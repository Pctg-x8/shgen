use std::{collections::BTreeMap, io::Write, rc::Rc};

use dsl::{
    Composite2, Constant, DescriptorImage2D, DescriptorImage3D, DescriptorRef, DescriptorType,
    DescriptorUniformBlock, Float4, Float4x4, GlobalInvocationIdRef, HalfFloatColor, InputStorage,
    Int2, Int3, LocationRef, Mutable, OutputStorage, PositionRef, ShaderAction, ShaderExpression,
    ShaderRefExpression, ShaderStorableExpression, StructMember, Type, TypeId, Uint3,
    VectorElementW, VectorElementX, VectorElementY, VectorElementZ,
};
use instruction_emitter::InstructionEmitter;
use spir::{ExecutionMode, ExecutionModel, Instruction};

use crate::spir::{AddressingModel, Capability, MemoryModel, ModuleBinaryHeader};

mod dsl;
mod instruction_emitter;
mod spir;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorRefDefinition {
    pub r#type: Descriptor,
    pub mutable: bool,
    pub use_half_float: bool,
}
impl DescriptorRefDefinition {
    pub const fn mutable(mut self) -> Self {
        self.mutable = true;

        self
    }

    pub const fn rgba16f(mut self) -> Self {
        self.use_half_float = true;

        self
    }
}
impl From<Descriptor> for DescriptorRefDefinition {
    fn from(value: Descriptor) -> Self {
        Self {
            r#type: value,
            mutable: false,
            use_half_float: false,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Descriptor {
    Array(Box<Descriptor>),
    UniformBlock,
    StorageBuffer,
    Image2D,
    Image3D,
}
impl Descriptor {
    pub const fn rw_decoratable(&self) -> bool {
        match self {
            Self::Array(a) => a.rw_decoratable(),
            Self::StorageBuffer | Self::Image2D | Self::Image3D => true,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DescriptorAttribute {
    Mutable,
    RGBA16F,
}

#[derive(Debug)]
pub struct ShaderInterface {
    descriptor_set_bounds: BTreeMap<u32, BTreeMap<u32, DescriptorRefDefinition>>,
    input_location_bounds: BTreeMap<u32, TypeId>,
    output_location_bounds: BTreeMap<u32, TypeId>,
}
impl ShaderInterface {
    pub const fn new() -> Self {
        Self {
            descriptor_set_bounds: BTreeMap::new(),
            input_location_bounds: BTreeMap::new(),
            output_location_bounds: BTreeMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct ComputeShaderContext {
    inputs: ShaderInterface,
    local_size_x: u32,
    local_size_y: Option<u32>,
    local_size_z: Option<u32>,
}
impl ComputeShaderContext {
    pub const fn new1(local_size_x: u32) -> Self {
        Self {
            inputs: ShaderInterface::new(),
            local_size_x,
            local_size_y: None,
            local_size_z: None,
        }
    }

    pub const fn new2(local_size_x: u32, local_size_y: u32) -> Self {
        Self {
            inputs: ShaderInterface::new(),
            local_size_x,
            local_size_y: Some(local_size_y),
            local_size_z: None,
        }
    }

    pub const fn new3(local_size_x: u32, local_size_y: u32, local_size_z: u32) -> Self {
        Self {
            inputs: ShaderInterface::new(),
            local_size_x,
            local_size_y: Some(local_size_y),
            local_size_z: Some(local_size_z),
        }
    }

    pub fn descriptor<T: DescriptorType>(&mut self, set: u32, bound: u32) -> DescriptorRef<T> {
        match self
            .inputs
            .descriptor_set_bounds
            .entry(set)
            .or_insert_with(BTreeMap::new)
            .entry(bound)
        {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::DEF);

                DescriptorRef {
                    set,
                    bound,
                    _ph: core::marker::PhantomData,
                }
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::DEF {
                    panic!("descriptor type mismatch before bound");
                }

                DescriptorRef {
                    set,
                    bound,
                    _ph: core::marker::PhantomData,
                }
            }
        }
    }

    pub const fn global_invocation_id(&self) -> impl ShaderExpression<Output = Uint3> {
        GlobalInvocationIdRef
    }

    pub fn combine_actions<Actions: ShaderAction>(
        self,
        actions: Actions,
    ) -> ComputeShader<Actions> {
        ComputeShader {
            inputs: self.inputs,
            local_size_x: self.local_size_x,
            local_size_y: self.local_size_y,
            local_size_z: self.local_size_z,
            actions,
        }
    }
}

#[derive(Debug)]
pub struct ComputeShader<Actions: ShaderAction> {
    #[allow(dead_code)]
    inputs: ShaderInterface,
    local_size_x: u32,
    local_size_y: Option<u32>,
    local_size_z: Option<u32>,
    actions: Actions,
}
impl<Actions: ShaderAction> ComputeShader<Actions> {
    pub fn emit_word_stream(&self) -> Vec<u32> {
        let mut ctx = InstructionEmitter::new();
        self.actions.emit(&mut ctx);
        let (epid, interface_ids, serialized1, bound) = ctx.serialize_instructions();
        let mut serialized = Vec::with_capacity(3 + serialized1.len());
        serialized.extend([
            Instruction::OpCapability(Capability::Shader),
            Instruction::OpMemoryModel(AddressingModel::Logical, MemoryModel::GLSL450),
            Instruction::OpEntryPoint {
                execution_model: ExecutionModel::GLCompute,
                func_id: epid,
                name: String::from("main"),
                interface: interface_ids,
            },
            Instruction::OpExecutionMode(
                epid,
                ExecutionMode::LocalSize(
                    self.local_size_x,
                    self.local_size_y.unwrap_or(1),
                    self.local_size_z.unwrap_or(1),
                ),
            ),
        ]);
        serialized.extend(serialized1);

        let mut word_stream = Vec::new();
        word_stream.extend(ModuleBinaryHeader::new(1, 0, 0, bound).into_words());
        for x in serialized {
            x.encode(&mut word_stream);
        }

        word_stream
    }

    pub fn export_module(
        &self,
        path: &(impl AsRef<std::path::Path> + ?Sized),
    ) -> std::io::Result<()> {
        let word_stream = self.emit_word_stream();

        let mut module = std::fs::File::options()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        module.write_all(unsafe {
            core::slice::from_raw_parts(word_stream.as_ptr() as *const u8, word_stream.len() * 4)
        })
    }
}

#[derive(Debug)]
pub enum FragCoordOrigin {
    LowerLeft,
    UpperLeft,
}

#[derive(Debug)]
pub struct FragmentShaderContext {
    inputs: ShaderInterface,
    frag_coord_origin: FragCoordOrigin,
}
impl FragmentShaderContext {
    pub const fn new() -> Self {
        Self {
            inputs: ShaderInterface::new(),
            frag_coord_origin: FragCoordOrigin::UpperLeft,
        }
    }

    pub fn frag_coord_origin_lower_left(&mut self) {
        self.frag_coord_origin = FragCoordOrigin::LowerLeft;
    }

    pub fn descriptor<T: DescriptorType>(&mut self, set: u32, bound: u32) -> DescriptorRef<T> {
        match self
            .inputs
            .descriptor_set_bounds
            .entry(set)
            .or_insert_with(BTreeMap::new)
            .entry(bound)
        {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::DEF);
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::DEF {
                    panic!("descriptor type mismatch before bound");
                }
            }
        }

        DescriptorRef {
            set,
            bound,
            _ph: core::marker::PhantomData,
        }
    }

    pub fn input_location<T: Type>(&mut self, location: u32) -> LocationRef<T, InputStorage> {
        match self.inputs.input_location_bounds.entry(location) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::id());
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::id() {
                    panic!("location type mismatch before bound");
                }
            }
        }

        LocationRef(location, core::marker::PhantomData)
    }

    pub fn output_location<T: Type>(&mut self, location: u32) -> LocationRef<T, OutputStorage> {
        match self.inputs.output_location_bounds.entry(location) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::id());
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::id() {
                    panic!("location type mismatch before bound");
                }
            }
        }

        LocationRef(location, core::marker::PhantomData)
    }

    pub fn combine_action<Action: ShaderAction>(self, action: Action) -> FragmentShader<Action> {
        FragmentShader {
            inputs: self.inputs,
            frag_coord_origin: self.frag_coord_origin,
            action,
        }
    }
}

#[derive(Debug)]
pub struct FragmentShader<Action: ShaderAction> {
    #[allow(dead_code)]
    inputs: ShaderInterface,
    frag_coord_origin: FragCoordOrigin,
    action: Action,
}
impl<Action: ShaderAction> FragmentShader<Action> {
    pub fn emit_word_stream(&self) -> Vec<u32> {
        let mut ctx = InstructionEmitter::new();
        self.action.emit(&mut ctx);
        let (epid, interface_ids, serialized1, bound) = ctx.serialize_instructions();
        let mut serialized = Vec::with_capacity(3 + serialized1.len());
        serialized.extend([
            Instruction::OpCapability(Capability::Shader),
            Instruction::OpMemoryModel(AddressingModel::Logical, MemoryModel::GLSL450),
            Instruction::OpEntryPoint {
                execution_model: ExecutionModel::Fragment,
                func_id: epid,
                name: String::from("main"),
                interface: interface_ids,
            },
            Instruction::OpExecutionMode(
                epid,
                match self.frag_coord_origin {
                    FragCoordOrigin::LowerLeft => ExecutionMode::OriginLowerLeft,
                    FragCoordOrigin::UpperLeft => ExecutionMode::OriginUpperLeft,
                },
            ),
        ]);
        serialized.extend(serialized1);

        let mut word_stream = Vec::new();
        word_stream.extend(ModuleBinaryHeader::new(1, 0, 0, bound).into_words());
        for x in serialized {
            x.encode(&mut word_stream);
        }

        word_stream
    }

    pub fn export_module(
        &self,
        path: &(impl AsRef<std::path::Path> + ?Sized),
    ) -> std::io::Result<()> {
        let word_stream = self.emit_word_stream();

        let mut module = std::fs::File::options()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        module.write_all(unsafe {
            core::slice::from_raw_parts(word_stream.as_ptr() as *const u8, word_stream.len() * 4)
        })
    }
}

#[derive(Debug)]
pub struct VertexShaderContext {
    inputs: ShaderInterface,
}
impl VertexShaderContext {
    pub const fn new() -> Self {
        Self {
            inputs: ShaderInterface::new(),
        }
    }

    pub fn descriptor<T: DescriptorType>(&mut self, set: u32, bound: u32) -> DescriptorRef<T> {
        match self
            .inputs
            .descriptor_set_bounds
            .entry(set)
            .or_insert_with(BTreeMap::new)
            .entry(bound)
        {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::DEF);
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::DEF {
                    panic!("descriptor type mismatch before bound");
                }
            }
        }

        DescriptorRef {
            set,
            bound,
            _ph: core::marker::PhantomData,
        }
    }

    pub fn input_location<T: Type>(&mut self, location: u32) -> LocationRef<T, InputStorage> {
        match self.inputs.input_location_bounds.entry(location) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::id());
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::id() {
                    panic!("location type mismatch before bound");
                }
            }
        }

        LocationRef(location, core::marker::PhantomData)
    }

    pub fn output_location<T: Type>(&mut self, location: u32) -> LocationRef<T, OutputStorage> {
        match self.inputs.output_location_bounds.entry(location) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(T::id());
            }
            std::collections::btree_map::Entry::Occupied(v) => {
                if *v.get() != T::id() {
                    panic!("location type mismatch before bound");
                }
            }
        }

        LocationRef(location, core::marker::PhantomData)
    }

    pub fn position(
        &self,
    ) -> impl ShaderStorableExpression<Output = Float4, StorageClass = OutputStorage> {
        PositionRef
    }

    pub fn combine_action<Action: ShaderAction>(self, action: Action) -> VertexShader<Action> {
        VertexShader {
            inputs: self.inputs,
            action,
        }
    }
}

#[derive(Debug)]
pub struct VertexShader<Action: ShaderAction> {
    #[allow(dead_code)]
    inputs: ShaderInterface,
    action: Action,
}
impl<Action: ShaderAction> VertexShader<Action> {
    pub fn emit_word_stream(&self) -> Vec<u32> {
        let mut ctx = InstructionEmitter::new();
        self.action.emit(&mut ctx);
        let (epid, interface_ids, serialized1, bound) = ctx.serialize_instructions();
        let mut serialized = Vec::with_capacity(3 + serialized1.len());
        serialized.extend([
            Instruction::OpCapability(Capability::Shader),
            Instruction::OpMemoryModel(AddressingModel::Logical, MemoryModel::GLSL450),
            Instruction::OpEntryPoint {
                execution_model: ExecutionModel::Vertex,
                func_id: epid,
                name: String::from("main"),
                interface: interface_ids,
            },
        ]);
        serialized.extend(serialized1);

        let mut word_stream = Vec::new();
        word_stream.extend(ModuleBinaryHeader::new(1, 0, 0, bound).into_words());
        for x in serialized {
            x.encode(&mut word_stream);
        }

        word_stream
    }

    pub fn export_module(
        &self,
        path: &(impl AsRef<std::path::Path> + ?Sized),
    ) -> std::io::Result<()> {
        let word_stream = self.emit_word_stream();

        let mut module = std::fs::File::options()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        module.write_all(unsafe {
            core::slice::from_raw_parts(word_stream.as_ptr() as *const u8, word_stream.len() * 4)
        })
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct SafeFloat(f32);
impl SafeFloat {
    pub fn new(v: f32) -> Option<Self> {
        if v.is_nan() {
            None
        } else {
            Some(Self(v))
        }
    }

    pub const unsafe fn new_unchecked(v: f32) -> Self {
        Self(v)
    }

    pub const fn value(self) -> f32 {
        self.0
    }
}
impl PartialEq for SafeFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl Eq for SafeFloat {}
impl core::hash::Hash for SafeFloat {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        unsafe { core::mem::transmute::<_, u32>(self.0).hash(state) }
    }
}

fn main() -> std::io::Result<()> {
    csh_accum2().export_module("accum2.spv")?;
    csh_accum3().export_module("accum3.spv")?;
    vertex_color_f().export_module("vertex_color_f.spv")?;
    vertex().export_module("vertex.spv")?;

    Ok(())
}

fn csh_accum2() -> ComputeShader<impl ShaderAction> {
    let mut ctx = ComputeShaderContext::new2(32, 32);
    let src = ctx.descriptor::<HalfFloatColor<DescriptorImage2D>>(0, 0);
    let dst = ctx.descriptor::<Mutable<HalfFloatColor<DescriptorImage2D>>>(1, 0);

    let p = Rc::new(
        ctx.global_invocation_id()
            .swizzle2(VectorElementX, VectorElementY)
            .cast::<Int2>(),
    );
    let v = src.load(p.clone()).add(dst.load(p.clone()));

    ctx.combine_actions(dst.store(p, v))
}

fn csh_accum3() -> ComputeShader<impl ShaderAction> {
    let mut ctx = ComputeShaderContext::new3(8, 8, 8);
    let src = ctx.descriptor::<HalfFloatColor<DescriptorImage3D>>(0, 0);
    let dst = ctx.descriptor::<Mutable<HalfFloatColor<DescriptorImage3D>>>(1, 0);

    let p = Rc::new(
        ctx.global_invocation_id()
            .swizzle3(VectorElementX, VectorElementY, VectorElementZ)
            .cast::<Int3>(),
    );
    let v = src.load(p.clone()).add(dst.load(p.clone()));

    ctx.combine_actions(dst.store(p, v))
}

fn vertex_color_f() -> FragmentShader<impl ShaderAction> {
    let mut ctx = FragmentShaderContext::new();
    let out = ctx.output_location::<Float4>(0);
    let vcol = ctx.input_location::<Float4>(0);

    let premul = Composite2(
        vcol.swizzle3(VectorElementX, VectorElementY, VectorElementZ),
        unsafe { SafeFloat::new_unchecked(1.0) },
    )
    .mul(vcol.swizzle1(VectorElementW));

    ctx.combine_action(out.store(premul))
}

fn vertex() -> VertexShader<impl ShaderAction> {
    let mut ctx = VertexShaderContext::new();
    let pos = ctx.input_location::<Float4>(0);
    let vcol = ctx.input_location::<Float4>(1);
    let ocol = ctx.output_location::<Float4>(0);
    let camera_uniform = ctx.descriptor::<DescriptorUniformBlock<CustomUniformStruct>>(0, 0);

    let ctrans = ocol.store(vcol);
    let vtrans = ctx
        .position()
        .store(CustomUniformStruct::mvp(camera_uniform).mul(pos));

    ctx.combine_action(vtrans.then(ctrans))
}

#[derive(Debug)]
struct CustomUniformStruct;
impl Type for CustomUniformStruct {
    fn id() -> TypeId {
        TypeId::Struct(vec![StructMember::of_type::<Float4x4>(0)])
    }
}
impl CustomUniformStruct {
    pub fn mvp(
        src: impl ShaderRefExpression<Output = impl DescriptorType<ValueType = Self>>,
    ) -> impl ShaderRefExpression<Output = Float4x4> {
        src.member::<Float4x4>(Constant::Uint(0))
    }
}
