use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
    rc::Rc,
    sync::Arc,
};

use spir::{
    AccessQualifier, Builtin, Decoration, ExecutionMode, ExecutionModel, FunctionControl, Id,
    ImageDepthFlag, ImageDimension, ImageFormat, ImageSamplingCompatibilityFlag, Instruction,
    PureResultInstruction, StorageClass,
};

use crate::spir::{AddressingModel, Capability, MemoryModel, ModuleBinaryHeader};

mod spir;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum TypeId {
    Void,
    Bool,
    Byte,
    HalfInt,
    Uint,
    Int,
    Float,
    Uint2,
    Int2,
    Float2,
    Uint3,
    Int3,
    Float3,
    Uint4,
    Int4,
    Float4,
    Float2x2,
    Float2x3,
    Float2x4,
    Float3x2,
    Float3x3,
    Float3x4,
    Float4x2,
    Float4x3,
    Float4x4,
    Uint2x2,
    Uint2x3,
    Uint2x4,
    Uint3x2,
    Uint3x3,
    Uint3x4,
    Uint4x2,
    Uint4x3,
    Uint4x4,
    Int2x2,
    Int2x3,
    Int2x4,
    Int3x2,
    Int3x3,
    Int3x4,
    Int4x2,
    Int4x3,
    Int4x4,
    Function(Vec<TypeId>, Box<TypeId>),
    Image {
        sampled_type: Box<TypeId>,
        dimension: ImageDimension,
        depth: ImageDepthFlag,
        arrayed: bool,
        multisampled: bool,
        sample_compat: ImageSamplingCompatibilityFlag,
        image_format: ImageFormat,
        access_qualifier: Option<AccessQualifier>,
    },
    Pointer(Box<TypeId>, StorageClass),
}
impl TypeId {
    fn make_instruction(&self, result: Id, ctx: &mut InstructionEmitter) -> Instruction {
        match self {
            Self::Void => Instruction::OpTypeVoid(result),
            Self::Bool => Instruction::OpTypeBool(result),
            Self::Byte => Instruction::OpTypeInt {
                result,
                bits: 8,
                signed: false,
            },
            Self::HalfInt => Instruction::OpTypeInt {
                result,
                bits: 16,
                signed: true,
            },
            Self::Uint => Instruction::OpTypeInt {
                result,
                bits: 32,
                signed: false,
            },
            Self::Int => Instruction::OpTypeInt {
                result,
                bits: 32,
                signed: true,
            },
            Self::Float => Instruction::OpTypeFloat(result, 32),
            Self::Uint2 => Instruction::OpTypeVector(result, ctx.type_id::<Uint>(), 2),
            Self::Uint3 => Instruction::OpTypeVector(result, ctx.type_id::<Uint>(), 3),
            Self::Uint4 => Instruction::OpTypeVector(result, ctx.type_id::<Uint>(), 4),
            Self::Int2 => Instruction::OpTypeVector(result, ctx.type_id::<Int>(), 2),
            Self::Int3 => Instruction::OpTypeVector(result, ctx.type_id::<Int>(), 3),
            Self::Int4 => Instruction::OpTypeVector(result, ctx.type_id::<Int>(), 4),
            Self::Float2 => Instruction::OpTypeVector(result, ctx.type_id::<Float>(), 2),
            Self::Float3 => Instruction::OpTypeVector(result, ctx.type_id::<Float>(), 3),
            Self::Float4 => Instruction::OpTypeVector(result, ctx.type_id::<Float>(), 4),
            Self::Float2x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float2>(), 2),
            Self::Float2x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float2>(), 3),
            Self::Float2x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float2>(), 4),
            Self::Float3x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float3>(), 2),
            Self::Float3x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float3>(), 3),
            Self::Float3x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float3>(), 4),
            Self::Float4x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float4>(), 2),
            Self::Float4x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float4>(), 3),
            Self::Float4x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Float4>(), 4),
            Self::Int2x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int2>(), 2),
            Self::Int2x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int2>(), 3),
            Self::Int2x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int2>(), 4),
            Self::Int3x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int3>(), 2),
            Self::Int3x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int3>(), 3),
            Self::Int3x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int3>(), 4),
            Self::Int4x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int4>(), 2),
            Self::Int4x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int4>(), 3),
            Self::Int4x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Int4>(), 4),
            Self::Uint2x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint2>(), 2),
            Self::Uint2x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint2>(), 3),
            Self::Uint2x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint2>(), 4),
            Self::Uint3x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint3>(), 2),
            Self::Uint3x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint3>(), 3),
            Self::Uint3x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint3>(), 4),
            Self::Uint4x2 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint4>(), 2),
            Self::Uint4x3 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint4>(), 3),
            Self::Uint4x4 => Instruction::OpTypeMatrix(result, ctx.type_id::<Uint4>(), 4),
            Self::Function(args, ret) => Instruction::OpTypeFunction {
                result,
                return_type: ctx.type_id_of_val(ret.as_ref().clone()),
                parameter_types: args
                    .iter()
                    .map(|tid| ctx.type_id_of_val(tid.clone()))
                    .collect(),
            },
            Self::Image {
                sampled_type,
                dimension,
                depth,
                arrayed,
                multisampled,
                sample_compat,
                image_format,
                access_qualifier,
            } => Instruction::OpTypeImage {
                result,
                sampled_type: ctx.type_id_of_val(sampled_type.as_ref().clone()),
                dim: *dimension,
                depth: *depth,
                arrayed: *arrayed,
                multisampled: *multisampled,
                sampling_cap: *sample_compat,
                format: *image_format,
                qualifier: access_qualifier.as_ref().copied(),
            },
            Self::Pointer(org_type, storage_class) => Instruction::OpTypePointer {
                result,
                storage_class: *storage_class,
                r#type: ctx.type_id_of_val(org_type.as_ref().clone()),
            },
        }
    }
}

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
    UniformBuffer,
    StorageBuffer,
    Image2D,
    Image3D,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DescriptorAttribute {
    Mutable,
    RGBA16F,
}

#[derive(Debug)]
pub struct ShaderInputs {
    descriptor_set_bounds: BTreeMap<u32, BTreeMap<u32, DescriptorRefDefinition>>,
    location_bounds: BTreeMap<u32, TypeId>,
}

#[derive(Debug)]
pub struct ComputeShaderContext {
    inputs: ShaderInputs,
    local_size_x: u32,
    local_size_y: Option<u32>,
    local_size_z: Option<u32>,
}
impl ComputeShaderContext {
    pub const fn new1(local_size_x: u32) -> Self {
        Self {
            inputs: ShaderInputs {
                descriptor_set_bounds: BTreeMap::new(),
                location_bounds: BTreeMap::new(),
            },
            local_size_x,
            local_size_y: None,
            local_size_z: None,
        }
    }

    pub const fn new2(local_size_x: u32, local_size_y: u32) -> Self {
        Self {
            inputs: ShaderInputs {
                descriptor_set_bounds: BTreeMap::new(),
                location_bounds: BTreeMap::new(),
            },
            local_size_x,
            local_size_y: Some(local_size_y),
            local_size_z: None,
        }
    }

    pub const fn new3(local_size_x: u32, local_size_y: u32, local_size_z: u32) -> Self {
        Self {
            inputs: ShaderInputs {
                descriptor_set_bounds: BTreeMap::new(),
                location_bounds: BTreeMap::new(),
            },
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
    inputs: ShaderInputs,
    local_size_x: u32,
    local_size_y: Option<u32>,
    local_size_z: Option<u32>,
    actions: Actions,
}
impl<Actions: ShaderAction> ComputeShader<Actions> {
    fn emit(&self, ctx: &mut InstructionEmitter) {
        self.actions.emit(ctx);
    }

    pub fn export_module(
        &self,
        path: &(impl AsRef<std::path::Path> + ?Sized),
    ) -> std::io::Result<()> {
        let mut ctx = InstructionEmitter::new();
        self.emit(&mut ctx);
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

        let mut module = std::fs::File::options()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let header = ModuleBinaryHeader::new(1, 0, 0, bound);
        module.write_all(header.as_bytes())?;
        let mut word_stream = Vec::new();
        for x in serialized {
            x.encode(&mut word_stream);
        }
        module.write_all(unsafe {
            core::slice::from_raw_parts(word_stream.as_ptr() as *const u8, word_stream.len() * 4)
        })
    }
}

pub trait Type: core::fmt::Debug {
    fn id() -> TypeId;
}
pub trait TypeAddRelation<Right: Type> {
    type Output: Type;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id;
}
pub trait VectorType: Type {
    type Element: Type;
}
pub trait VectorTypeFamily {
    type Vector2: Type;
    type Vector3: Type;
    type Vector4: Type;
}
pub trait ImageType: Type {
    type SampledType: Type + VectorTypeFamily;
    const DIMENSION: ImageDimension;
    const DEPTH: ImageDepthFlag;
    const ARRAYED: bool;
    const MULTISAMPLED: bool;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag;
    const FORMAT: ImageFormat;
    const QUALIFIER: Option<AccessQualifier> = None;
}

fn image_type_id<T: ImageType>() -> TypeId {
    TypeId::Image {
        sampled_type: Box::new(<T::SampledType as Type>::id()),
        dimension: T::DIMENSION,
        depth: T::DEPTH,
        arrayed: T::ARRAYED,
        multisampled: T::MULTISAMPLED,
        sample_compat: T::SAMPLE_COMPAT,
        image_format: T::FORMAT,
        access_qualifier: T::QUALIFIER,
    }
}

pub trait TypeList: core::fmt::Debug {
    fn id_list() -> Vec<TypeId>;
}
impl TypeList for () {
    fn id_list() -> Vec<TypeId> {
        vec![]
    }
}
impl<A: Type> TypeList for (A,) {
    fn id_list() -> Vec<TypeId> {
        vec![A::id()]
    }
}
impl<A: Type, B: Type> TypeList for (A, B) {
    fn id_list() -> Vec<TypeId> {
        vec![A::id(), B::id()]
    }
}
impl<A: Type, B: Type, C: Type> TypeList for (A, B, C) {
    fn id_list() -> Vec<TypeId> {
        vec![A::id(), B::id(), C::id()]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CallableType<Args: TypeList, Return: Type>(core::marker::PhantomData<(Args, Return)>);
impl<Args: TypeList, Return: Type> Type for CallableType<Args, Return> {
    fn id() -> TypeId {
        TypeId::Function(Args::id_list(), Box::new(Return::id()))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Image2D<SampledType: Type + VectorTypeFamily>(core::marker::PhantomData<SampledType>);
impl<SampledType: Type + VectorTypeFamily> ImageType for Image2D<SampledType> {
    type SampledType = SampledType;
    const DIMENSION: ImageDimension = ImageDimension::Two;
    const DEPTH: ImageDepthFlag = ImageDepthFlag::NonDepth;
    const ARRAYED: bool = false;
    const MULTISAMPLED: bool = false;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag = ImageSamplingCompatibilityFlag::ReadWrite;
    const FORMAT: ImageFormat = ImageFormat::Rgba8;
}
impl<SampledType: Type + VectorTypeFamily> Type for Image2D<SampledType> {
    fn id() -> TypeId {
        image_type_id::<Self>()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Image3D<SampledType: Type + VectorTypeFamily>(core::marker::PhantomData<SampledType>);
impl<SampledType: Type + VectorTypeFamily> ImageType for Image3D<SampledType> {
    type SampledType = SampledType;
    const DIMENSION: ImageDimension = ImageDimension::Three;
    const DEPTH: ImageDepthFlag = ImageDepthFlag::NonDepth;
    const ARRAYED: bool = false;
    const MULTISAMPLED: bool = false;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag = ImageSamplingCompatibilityFlag::ReadWrite;
    const FORMAT: ImageFormat = ImageFormat::Rgba8;
}
impl<SampledType: Type + VectorTypeFamily> Type for Image3D<SampledType> {
    fn id() -> TypeId {
        image_type_id::<Self>()
    }
}

pub trait StorageClassMarker: core::fmt::Debug {
    const VALUE: StorageClass;
}

#[derive(Debug, Clone, Copy)]
pub struct UniformConstantStorage;
impl StorageClassMarker for UniformConstantStorage {
    const VALUE: StorageClass = StorageClass::UniformConstant;
}

#[derive(Debug, Clone, Copy)]
pub struct InputStorage;
impl StorageClassMarker for InputStorage {
    const VALUE: StorageClass = StorageClass::Input;
}

#[derive(Clone, Copy, Debug)]
pub struct Pointer<T: Type, C: StorageClassMarker>(core::marker::PhantomData<(T, C)>);
impl<T: Type, C: StorageClassMarker> Type for Pointer<T, C> {
    fn id() -> TypeId {
        TypeId::Pointer(Box::new(T::id()), C::VALUE)
    }
}

type UniformConstantPointer<T> = Pointer<T, UniformConstantStorage>;

#[derive(Clone, Copy, Debug)]
pub struct Void;
impl Type for Void {
    fn id() -> TypeId {
        TypeId::Void
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bool;
impl Type for Bool {
    fn id() -> TypeId {
        TypeId::Bool
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Float;
impl Type for Float {
    fn id() -> TypeId {
        TypeId::Float
    }
}
impl TypeAddRelation<Float> for Float {
    type Output = Float;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Float>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl VectorTypeFamily for Float {
    type Vector2 = Float2;
    type Vector3 = Float3;
    type Vector4 = Float4;
}

#[derive(Clone, Copy, Debug)]
pub struct Uint;
impl Type for Uint {
    fn id() -> TypeId {
        TypeId::Uint
    }
}
impl TypeAddRelation<Uint> for Uint {
    type Output = Uint;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Uint>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl VectorTypeFamily for Uint {
    type Vector2 = Uint2;
    type Vector3 = Uint3;
    type Vector4 = Uint4;
}

#[derive(Clone, Copy, Debug)]
pub struct Int;
impl Type for Int {
    fn id() -> TypeId {
        TypeId::Int
    }
}
impl TypeAddRelation<Int> for Int {
    type Output = Int;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Int>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl VectorTypeFamily for Int {
    type Vector2 = Int2;
    type Vector3 = Int3;
    type Vector4 = Int4;
}

#[derive(Clone, Copy, Debug)]
pub struct Float2;
impl Type for Float2 {
    fn id() -> TypeId {
        TypeId::Float2
    }
}
impl VectorType for Float2 {
    type Element = Float;
}
impl TypeAddRelation<Float2> for Float2 {
    type Output = Float2;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Float2>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Float3;
impl Type for Float3 {
    fn id() -> TypeId {
        TypeId::Float3
    }
}
impl VectorType for Float3 {
    type Element = Float;
}
impl TypeAddRelation<Float3> for Float3 {
    type Output = Float3;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Float3>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Float4;
impl Type for Float4 {
    fn id() -> TypeId {
        TypeId::Float4
    }
}
impl VectorType for Float4 {
    type Element = Float;
}
impl TypeAddRelation<Float4> for Float4 {
    type Output = Float4;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Float4>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int2;
impl Type for Int2 {
    fn id() -> TypeId {
        TypeId::Int2
    }
}
impl VectorType for Int2 {
    type Element = Int;
}
impl TypeAddRelation<Int2> for Int2 {
    type Output = Int2;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Int2>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int3;
impl Type for Int3 {
    fn id() -> TypeId {
        TypeId::Int3
    }
}
impl VectorType for Int3 {
    type Element = Int;
}
impl TypeAddRelation<Int3> for Int3 {
    type Output = Int3;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Int3>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Int4;
impl Type for Int4 {
    fn id() -> TypeId {
        TypeId::Int4
    }
}
impl VectorType for Int4 {
    type Element = Int;
}
impl TypeAddRelation<Int4> for Int4 {
    type Output = Int4;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Int4>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Uint2;
impl Type for Uint2 {
    fn id() -> TypeId {
        TypeId::Uint2
    }
}
impl VectorType for Uint2 {
    type Element = Uint;
}
impl TypeAddRelation<Uint2> for Uint2 {
    type Output = Uint2;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Uint2>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Uint3;
impl Type for Uint3 {
    fn id() -> TypeId {
        TypeId::Uint3
    }
}
impl VectorType for Uint3 {
    type Element = Uint;
}
impl TypeAddRelation<Uint3> for Uint3 {
    type Output = Uint3;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Uint3>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Uint4;
impl Type for Uint4 {
    fn id() -> TypeId {
        TypeId::Uint4
    }
}
impl VectorType for Uint4 {
    type Element = Uint;
}
impl TypeAddRelation<Uint4> for Uint4 {
    type Output = Uint4;

    fn emit(left: Id, right: Id, ctx: &mut InstructionEmitter) -> Id {
        let rty = ctx.type_id::<Uint4>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    True,
    False,
    Null(TypeId),
    Float(SafeFloat),
    Uint(u32),
    Int(i32),
    Float2(SafeFloat, SafeFloat),
    Float3(SafeFloat, SafeFloat, SafeFloat),
    Float4(SafeFloat, SafeFloat, SafeFloat, SafeFloat),
    Uint2(u32, u32),
    Uint3(u32, u32, u32),
    Uint4(u32, u32, u32, u32),
    Int2(i32, i32),
    Int3(i32, i32, i32),
    Int4(i32, i32, i32, i32),
}
impl Constant {
    pub fn make_instruction(&self, result: Id, ctx: &mut InstructionEmitter) -> Instruction {
        match self {
            Self::True => Instruction::OpConstantTrue {
                result_type: ctx.type_id::<Bool>(),
                result,
            },
            Self::False => Instruction::OpConstantFalse {
                result_type: ctx.type_id::<Bool>(),
                result,
            },
            Self::Null(t) => Instruction::OpConstantNull {
                result_type: ctx.type_id_of_val(t.clone()),
                result,
            },
            &Self::Float(f) => Instruction::OpConstant {
                result_type: ctx.type_id::<Float>(),
                result,
                value: vec![unsafe { core::mem::transmute(f.value()) }],
            },
            &Self::Uint(v) => Instruction::OpConstant {
                result_type: ctx.type_id::<Uint>(),
                result,
                value: vec![v],
            },
            &Self::Int(v) => Instruction::OpConstant {
                result_type: ctx.type_id::<Int>(),
                result,
                value: vec![unsafe { core::mem::transmute(v) }],
            },
            &Self::Float2(x, y) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Float2>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Float(x)),
                    ctx.constant_id(Self::Float(y)),
                ],
            },
            &Self::Float3(x, y, z) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Float3>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Float(x)),
                    ctx.constant_id(Self::Float(y)),
                    ctx.constant_id(Self::Float(z)),
                ],
            },
            &Self::Float4(x, y, z, w) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Float4>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Float(x)),
                    ctx.constant_id(Self::Float(y)),
                    ctx.constant_id(Self::Float(z)),
                    ctx.constant_id(Self::Float(w)),
                ],
            },
            &Self::Uint2(x, y) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Uint2>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Uint(x)),
                    ctx.constant_id(Self::Uint(y)),
                ],
            },
            &Self::Uint3(x, y, z) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Uint3>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Uint(x)),
                    ctx.constant_id(Self::Uint(y)),
                    ctx.constant_id(Self::Uint(z)),
                ],
            },
            &Self::Uint4(x, y, z, w) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Uint4>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Uint(x)),
                    ctx.constant_id(Self::Uint(y)),
                    ctx.constant_id(Self::Uint(z)),
                    ctx.constant_id(Self::Uint(w)),
                ],
            },
            &Self::Int2(x, y) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Int2>(),
                result,
                consituents: vec![ctx.constant_id(Self::Int(x)), ctx.constant_id(Self::Int(y))],
            },
            &Self::Int3(x, y, z) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Int3>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Int(x)),
                    ctx.constant_id(Self::Int(y)),
                    ctx.constant_id(Self::Int(z)),
                ],
            },
            &Self::Int4(x, y, z, w) => Instruction::OpConstantComposite {
                result_type: ctx.type_id::<Int4>(),
                result,
                consituents: vec![
                    ctx.constant_id(Self::Int(x)),
                    ctx.constant_id(Self::Int(y)),
                    ctx.constant_id(Self::Int(z)),
                    ctx.constant_id(Self::Int(w)),
                ],
            },
        }
    }
}

pub trait ShaderAction: core::fmt::Debug {
    fn emit(&self, ctx: &mut InstructionEmitter);

    fn then<A: ShaderAction>(self, next: A) -> ShaderActionChain<Self, A>
    where
        Self: Sized,
    {
        ShaderActionChain {
            current: self,
            next,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShaderActionChain<T: ShaderAction, Next: ShaderAction> {
    pub current: T,
    pub next: Next,
}
impl<T: ShaderAction, Next: ShaderAction> ShaderAction for ShaderActionChain<T, Next> {
    fn emit(&self, ctx: &mut InstructionEmitter) {
        self.current.emit(ctx);
        self.next.emit(ctx);
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct ShaderExpressionAction<Expr: ShaderExpression<Output = Void>>(pub Expr);
impl<Expr: ShaderExpression<Output = Void>> ShaderAction for ShaderExpressionAction<Expr> {
    fn emit(&self, ctx: &mut InstructionEmitter) {
        self.0.emit(ctx);
    }
}
impl<Expr: ShaderExpression<Output = Void>> core::fmt::Debug for ShaderExpressionAction<Expr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Expr::fmt(&self.0, f)
    }
}

pub trait ShaderExpression: core::fmt::Debug {
    type Output: Type;

    /// returns resulting id of the expression
    fn emit(&self, ctx: &mut InstructionEmitter) -> Id;

    fn into_action(self) -> ShaderExpressionAction<Self>
    where
        Self: Sized + ShaderExpression<Output = Void>,
    {
        ShaderExpressionAction(self)
    }

    fn apply<Args>(self, args: Args) -> Apply<Self, Args>
    where
        Self: Sized,
        Args: ShaderExpressionList,
    {
        Apply(self, args)
    }

    fn add<Right>(self, right: Right) -> Add<Self, Right>
    where
        Self: Sized,
        Right: ShaderExpression,
        Self::Output: TypeAddRelation<Right::Output>,
    {
        Add(self, right)
    }

    fn cast<T>(self) -> Cast<T, Self>
    where
        Self: Sized,
        T: Type,
    {
        Cast(self, core::marker::PhantomData)
    }

    fn swizzle1<X>(self, _x: X) -> VectorSwizzle1<Self, X>
    where
        Self: Sized,
        X: VectorElementOf<Self::Output>,
    {
        VectorSwizzle1(self, core::marker::PhantomData)
    }

    fn swizzle2<X, Y>(self, _x: X, _y: Y) -> VectorSwizzle2<Self, X, Y>
    where
        Self: Sized,
        X: VectorElementOf<Self::Output>,
        Y: VectorElementOf<Self::Output>,
    {
        VectorSwizzle2(self, core::marker::PhantomData)
    }

    fn swizzle3<X, Y, Z>(self, _x: X, _y: Y, _z: Z) -> VectorSwizzle3<Self, X, Y, Z>
    where
        Self: Sized,
        X: VectorElementOf<Self::Output>,
        Y: VectorElementOf<Self::Output>,
        Z: VectorElementOf<Self::Output>,
    {
        VectorSwizzle3(self, core::marker::PhantomData)
    }

    fn swizzle4<X, Y, Z, W>(self, _x: X, _y: Y, _z: Z, _w: W) -> VectorSwizzle4<Self, X, Y, Z, W>
    where
        Self: Sized,
        X: VectorElementOf<Self::Output>,
        Y: VectorElementOf<Self::Output>,
        Z: VectorElementOf<Self::Output>,
        W: VectorElementOf<Self::Output>,
    {
        VectorSwizzle4(self, core::marker::PhantomData)
    }
}

impl<T: ShaderExpression> ShaderExpression for Rc<T> {
    type Output = T::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        T::emit(&self, ctx)
    }
}
impl<T: ShaderExpression> ShaderExpression for Arc<T> {
    type Output = T::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        T::emit(&self, ctx)
    }
}

pub trait ShaderExpressionList: core::fmt::Debug {
    type Type: TypeList;

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<Id>;
}
impl<T> ShaderExpressionList for (T,)
where
    T: ShaderExpression,
{
    type Type = (T::Output,);

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<Id> {
        vec![self.0.emit(ctx)]
    }
}
impl<A, B> ShaderExpressionList for (A, B)
where
    A: ShaderExpression,
    B: ShaderExpression,
{
    type Type = (A::Output, B::Output);

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<Id> {
        vec![self.0.emit(ctx), self.1.emit(ctx)]
    }
}
impl<A, B, C> ShaderExpressionList for (A, B, C)
where
    A: ShaderExpression,
    B: ShaderExpression,
    C: ShaderExpression,
{
    type Type = (A::Output, B::Output, C::Output);

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<Id> {
        vec![self.0.emit(ctx), self.1.emit(ctx), self.2.emit(ctx)]
    }
}

pub trait BuiltinRef {
    const ID: Builtin;
    type StorageClass: StorageClassMarker;
    type ValueType: Type;
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalInvocationIdRef;
impl BuiltinRef for GlobalInvocationIdRef {
    const ID: Builtin = Builtin::GlobalInvocationId;
    type StorageClass = InputStorage;
    type ValueType = Uint3;
}
impl ShaderExpression for GlobalInvocationIdRef {
    type Output = Uint3;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let vid = ctx.builtin_var_id::<Self>();
        let rty = ctx.type_id::<Self::Output>();

        ctx.load(rty, vid)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IntrinsicImageLoad<
    Source: ShaderExpression,
    Location: ShaderExpression<Output = <Source::Output as ImageLoadable>::Location>,
>(Source, Location)
where
    Source::Output: ImageLoadable;
impl<
        Source: ShaderExpression,
        Location: ShaderExpression<Output = <Source::Output as ImageLoadable>::Location>,
    > ShaderExpression for IntrinsicImageLoad<Source, Location>
where
    Source::Output: ImageLoadable,
{
    type Output = <<Source::Output as ImageLoadable>::SampledType as VectorTypeFamily>::Vector4;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let src = self.0.emit(ctx);
        let loc = self.1.emit(ctx);
        let result_ty = ctx.type_id::<Self::Output>();

        let rid = ctx.alloc_ret_id();
        ctx.add_main_instruction(Instruction::OpImageRead {
            result_type: result_ty,
            result: rid,
            image: src,
            coordinate: loc,
            operands: None,
        });
        rid
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IntrinsicImageStore<
    Destination: ShaderExpression,
    Location: ShaderExpression<Output = <Destination::Output as ImageLoadable>::Location>,
    Value: ShaderExpression<
        Output = <<Destination::Output as ImageLoadable>::SampledType as VectorTypeFamily>::Vector4,
    >,
>(Destination, Location, Value)
where
    Destination::Output: ImageStorable;
impl<
        Destination: ShaderExpression,
        Location: ShaderExpression<Output = <Destination::Output as ImageLoadable>::Location>,
        Value: ShaderExpression<Output = <<Destination::Output as ImageLoadable>::SampledType as VectorTypeFamily>::Vector4>,
    > ShaderAction for IntrinsicImageStore<Destination, Location, Value>
where
    Destination::Output: ImageStorable,
{
    fn emit(&self, ctx: &mut InstructionEmitter) {
        let dst = self.0.emit(ctx);
        let loc = self.1.emit(ctx);
        let val = self.2.emit(ctx);

        ctx.add_main_instruction(Instruction::OpImageWrite {
            image: dst,
            coordinate: loc,
            texel: val,
            operands: None,
        });
    }
}

pub trait DescriptorType: Type {
    const DEF: DescriptorRefDefinition;
}

pub trait ImageLoadable {
    type SampledType: Type + VectorTypeFamily;
    type Location: Type;
}
pub trait ImageStorable: ImageLoadable {}

#[derive(Clone, Copy, Debug)]
pub struct DescriptorImage2D<SampledType: Type + VectorTypeFamily = Float>(
    core::marker::PhantomData<SampledType>,
);
impl<SampledType: Type + VectorTypeFamily> ImageType for DescriptorImage2D<SampledType> {
    type SampledType = SampledType;
    const DIMENSION: ImageDimension = ImageDimension::Two;
    const DEPTH: ImageDepthFlag = ImageDepthFlag::NonDepth;
    const ARRAYED: bool = false;
    const MULTISAMPLED: bool = false;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag = ImageSamplingCompatibilityFlag::ReadWrite;
    const FORMAT: ImageFormat = ImageFormat::Rgba8;
}
impl<SampledType: Type + VectorTypeFamily> Type for DescriptorImage2D<SampledType> {
    fn id() -> TypeId {
        image_type_id::<Self>()
    }
}
impl<SampledType: Type + VectorTypeFamily> DescriptorType for DescriptorImage2D<SampledType> {
    const DEF: DescriptorRefDefinition = DescriptorRefDefinition {
        r#type: Descriptor::Image2D,
        mutable: false,
        use_half_float: false,
    };
}
impl<SampledType: Type + VectorTypeFamily> ImageLoadable for DescriptorImage2D<SampledType> {
    type SampledType = SampledType;
    type Location = Int2;
}

#[derive(Clone, Copy, Debug)]
pub struct DescriptorImage3D<SampledType: Type + VectorTypeFamily = Float>(
    core::marker::PhantomData<SampledType>,
);
impl<SampledType: Type + VectorTypeFamily> ImageType for DescriptorImage3D<SampledType> {
    type SampledType = SampledType;
    const DIMENSION: ImageDimension = ImageDimension::Three;
    const DEPTH: ImageDepthFlag = ImageDepthFlag::NonDepth;
    const ARRAYED: bool = false;
    const MULTISAMPLED: bool = false;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag = ImageSamplingCompatibilityFlag::ReadWrite;
    const FORMAT: ImageFormat = ImageFormat::Rgba8;
}
impl<SampledType: Type + VectorTypeFamily> Type for DescriptorImage3D<SampledType> {
    fn id() -> TypeId {
        image_type_id::<Self>()
    }
}
impl<SampledType: Type + VectorTypeFamily> DescriptorType for DescriptorImage3D<SampledType> {
    const DEF: DescriptorRefDefinition = DescriptorRefDefinition {
        r#type: Descriptor::Image3D,
        mutable: false,
        use_half_float: false,
    };
}
impl<SampledType: Type + VectorTypeFamily> ImageLoadable for DescriptorImage3D<SampledType> {
    type SampledType = SampledType;
    type Location = Int3;
}

#[derive(Clone, Copy, Debug)]
pub struct Mutable<T>(pub T);
impl<T: ImageType> ImageType for Mutable<T> {
    type SampledType = T::SampledType;
    const DIMENSION: ImageDimension = T::DIMENSION;
    const DEPTH: ImageDepthFlag = T::DEPTH;
    const ARRAYED: bool = T::ARRAYED;
    const MULTISAMPLED: bool = T::MULTISAMPLED;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag = T::SAMPLE_COMPAT;
    const FORMAT: ImageFormat = T::FORMAT;
}
impl<T: ImageType> Type for Mutable<T> {
    fn id() -> TypeId {
        image_type_id::<Self>()
    }
}
impl<T: DescriptorType + ImageType> DescriptorType for Mutable<T> {
    const DEF: DescriptorRefDefinition = T::DEF.mutable();
}
impl<T: ImageLoadable> ImageLoadable for Mutable<T> {
    type SampledType = T::SampledType;
    type Location = T::Location;
}
impl<T: ImageLoadable> ImageStorable for Mutable<T> {}

#[derive(Clone, Copy, Debug)]
pub struct HalfFloatColor<T>(pub T);
impl<T: ImageType> ImageType for HalfFloatColor<T> {
    type SampledType = T::SampledType;
    const DIMENSION: ImageDimension = T::DIMENSION;
    const DEPTH: ImageDepthFlag = T::DEPTH;
    const ARRAYED: bool = T::ARRAYED;
    const MULTISAMPLED: bool = T::MULTISAMPLED;
    const SAMPLE_COMPAT: ImageSamplingCompatibilityFlag = T::SAMPLE_COMPAT;
    const FORMAT: ImageFormat = ImageFormat::Rgba16f;
}
impl<T: ImageType> Type for HalfFloatColor<T> {
    fn id() -> TypeId {
        image_type_id::<Self>()
    }
}
impl<T: DescriptorType + ImageType> DescriptorType for HalfFloatColor<T> {
    const DEF: DescriptorRefDefinition = T::DEF.rgba16f();
}
impl<T: ImageLoadable> ImageLoadable for HalfFloatColor<T> {
    type SampledType = T::SampledType;
    type Location = T::Location;
}
impl<T: ImageStorable> ImageStorable for HalfFloatColor<T> {}

pub trait ValueType {
    const TYPE_ID: TypeId;
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorRefKey {
    set: u32,
    bound: u32,
    decl: DescriptorRefDefinition,
}

#[derive(Clone, Copy, Debug)]
pub struct DescriptorRef<T: DescriptorType> {
    set: u32,
    bound: u32,
    _ph: core::marker::PhantomData<T>,
}
impl<T: DescriptorType> DescriptorRef<T> {
    fn make_key(&self) -> DescriptorRefKey {
        DescriptorRefKey {
            set: self.set,
            bound: self.bound,
            decl: T::DEF,
        }
    }
}
impl<T: DescriptorType> ShaderExpression for DescriptorRef<T> {
    type Output = T;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let did = ctx.descriptor_var_id(&self);
        let rty = ctx.type_id::<T>();

        ctx.load(rty, did)
    }
}
impl<T: DescriptorType> DescriptorRef<T> {
    pub fn load<P>(
        self,
        pos: P,
    ) -> impl ShaderExpression<Output = <T::SampledType as VectorTypeFamily>::Vector4>
    where
        T: ImageLoadable,
        P: ShaderExpression<Output = T::Location>,
    {
        IntrinsicImageLoad(self, pos)
    }

    pub fn store<P, V>(self, pos: P, value: V) -> impl ShaderAction
    where
        T: ImageStorable,
        P: ShaderExpression<Output = T::Location>,
        V: ShaderExpression<Output = <T::SampledType as VectorTypeFamily>::Vector4>,
    {
        IntrinsicImageStore(self, pos, value)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocationRef<T: Type>(u32, core::marker::PhantomData<T>);
impl<T: Type> ShaderExpression for LocationRef<T> {
    type Output = T;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let tid = ctx.type_id::<T>();

        // TODO: emit instruction
        0
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CastStrategy {
    Bitwise,
    ConvertFToS,
    ConvertSToF,
}
pub trait CastableTo<T> {
    const STRATEGY: CastStrategy;
}
impl CastableTo<Int> for Uint {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Uint> for Int {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Int2> for Uint2 {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Uint2> for Int2 {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Int3> for Uint3 {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Uint3> for Int3 {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Int4> for Uint4 {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}
impl CastableTo<Uint4> for Int4 {
    const STRATEGY: CastStrategy = CastStrategy::Bitwise;
}

#[derive(Debug, Clone)]
pub struct Cast<T: Type, Source: ShaderExpression>(Source, core::marker::PhantomData<T>);
impl<T: Type, Source: ShaderExpression> ShaderExpression for Cast<T, Source>
where
    Source::Output: CastableTo<T>,
{
    type Output = T;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let source = self.0.emit(ctx);

        match <Source::Output as CastableTo<T>>::STRATEGY {
            CastStrategy::Bitwise => {
                let rty = ctx.type_id::<T>();

                ctx.pure_result_instruction(PureResultInstruction::OpBitcast {
                    result_type: rty,
                    operand: source,
                })
            }
            _ => todo!("casting strategy"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Apply<Callable: ShaderExpression, Args: ShaderExpressionList>(Callable, Args);
impl<
        R: Type,
        Callable: ShaderExpression<Output = CallableType<Args::Type, R>>,
        Args: ShaderExpressionList,
    > ShaderExpression for Apply<Callable, Args>
{
    type Output = R;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let callable = self.0.emit(ctx);
        let args = self.1.emit_all(ctx);
        let result_ty = ctx.type_id::<R>();

        // TODO: emit instruction
        0
    }
}

#[derive(Debug, Clone)]
pub struct Add<Left: ShaderExpression, Right: ShaderExpression>(Left, Right);
impl<Left: ShaderExpression, Right: ShaderExpression> ShaderExpression for Add<Left, Right>
where
    Left::Output: TypeAddRelation<Right::Output>,
{
    type Output = <Left::Output as TypeAddRelation<Right::Output>>::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let lhs = self.0.emit(ctx);
        let rhs = self.1.emit(ctx);

        <Left::Output as TypeAddRelation<Right::Output>>::emit(lhs, rhs, ctx)
    }
}

pub trait VectorElement: core::fmt::Debug {
    const INDEX: u32;
}
pub trait VectorElementOf<T: Type>: VectorElement {}

#[derive(Debug, Clone, Copy)]
pub struct VectorElementX;
impl VectorElement for VectorElementX {
    const INDEX: u32 = 0;
}
impl VectorElementOf<Float2> for VectorElementX {}
impl VectorElementOf<Float4> for VectorElementX {}
impl VectorElementOf<Int2> for VectorElementX {}
impl VectorElementOf<Uint3> for VectorElementX {}
#[derive(Debug, Clone, Copy)]
pub struct VectorElementY;
impl VectorElement for VectorElementY {
    const INDEX: u32 = 1;
}
impl VectorElementOf<Float2> for VectorElementY {}
impl VectorElementOf<Float4> for VectorElementY {}
impl VectorElementOf<Int2> for VectorElementY {}
impl VectorElementOf<Uint3> for VectorElementY {}
#[derive(Debug, Clone, Copy)]
pub struct VectorElementZ;
impl VectorElement for VectorElementZ {
    const INDEX: u32 = 2;
}
impl VectorElementOf<Float4> for VectorElementZ {}
impl VectorElementOf<Uint3> for VectorElementZ {}

#[derive(Debug, Clone, Copy)]
pub struct VectorElementW;
impl VectorElement for VectorElementW {
    const INDEX: u32 = 3;
}
impl VectorElementOf<Float4> for VectorElementW {}

#[derive(Debug, Clone)]
pub struct VectorSwizzle1<Source: ShaderExpression, E: VectorElementOf<Source::Output>>(
    Source,
    core::marker::PhantomData<E>,
);
impl<Source: ShaderExpression, E: VectorElementOf<Source::Output>> ShaderExpression
    for VectorSwizzle1<Source, E>
where
    Source::Output: VectorType,
{
    type Output = <Source::Output as VectorType>::Element;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let src = self.0.emit(ctx);

        // Note: never identical (vector-type <> scalar-type)

        let rty = ctx.type_id::<Self::Output>();
        ctx.pure_result_instruction(PureResultInstruction::OpVectorShuffle {
            result_type: rty,
            vector1: src,
            vector2: src,
            components: vec![E::INDEX],
        })
    }
}
#[derive(Debug, Clone)]
pub struct VectorSwizzle2<
    Source: ShaderExpression,
    E: VectorElementOf<Source::Output>,
    E2: VectorElementOf<Source::Output>,
>(Source, core::marker::PhantomData<(E, E2)>);
impl<
        Source: ShaderExpression,
        E: VectorElementOf<Source::Output>,
        E2: VectorElementOf<Source::Output>,
    > ShaderExpression for VectorSwizzle2<Source, E, E2>
where
    Source::Output: VectorType,
    <Source::Output as VectorType>::Element: VectorTypeFamily,
{
    type Output = <<Source::Output as VectorType>::Element as VectorTypeFamily>::Vector2;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let src = self.0.emit(ctx);

        if <Self::Output as Type>::id() == <Source::Output as Type>::id()
            && [E::INDEX, E2::INDEX] == [0, 1]
        {
            // identical, no operation emitted
            return src;
        }

        let rty = ctx.type_id::<Self::Output>();
        ctx.pure_result_instruction(PureResultInstruction::OpVectorShuffle {
            result_type: rty,
            vector1: src,
            vector2: src,
            components: vec![E::INDEX, E2::INDEX],
        })
    }
}
#[derive(Debug, Clone)]
pub struct VectorSwizzle3<
    Source: ShaderExpression,
    E: VectorElementOf<Source::Output>,
    E2: VectorElementOf<Source::Output>,
    E3: VectorElementOf<Source::Output>,
>(Source, core::marker::PhantomData<(E, E2, E3)>);
impl<
        Source: ShaderExpression,
        E: VectorElementOf<Source::Output>,
        E2: VectorElementOf<Source::Output>,
        E3: VectorElementOf<Source::Output>,
    > ShaderExpression for VectorSwizzle3<Source, E, E2, E3>
where
    Source::Output: VectorType,
    <Source::Output as VectorType>::Element: VectorTypeFamily,
{
    type Output = <<Source::Output as VectorType>::Element as VectorTypeFamily>::Vector3;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let src = self.0.emit(ctx);

        if <Self::Output as Type>::id() == <Source::Output as Type>::id()
            && [E::INDEX, E2::INDEX, E3::INDEX] == [0, 1, 2]
        {
            // identical, no operation emitted
            return src;
        }

        let rty = ctx.type_id::<Self::Output>();
        ctx.pure_result_instruction(PureResultInstruction::OpVectorShuffle {
            result_type: rty,
            vector1: src,
            vector2: src,
            components: vec![E::INDEX, E2::INDEX, E3::INDEX],
        })
    }
}
#[derive(Debug, Clone)]
pub struct VectorSwizzle4<
    Source: ShaderExpression,
    E: VectorElementOf<Source::Output>,
    E2: VectorElementOf<Source::Output>,
    E3: VectorElementOf<Source::Output>,
    E4: VectorElementOf<Source::Output>,
>(Source, core::marker::PhantomData<(E, E2, E3, E4)>);
impl<
        Source: ShaderExpression,
        E: VectorElementOf<Source::Output>,
        E2: VectorElementOf<Source::Output>,
        E3: VectorElementOf<Source::Output>,
        E4: VectorElementOf<Source::Output>,
    > ShaderExpression for VectorSwizzle4<Source, E, E2, E3, E4>
where
    Source::Output: VectorType,
    <Source::Output as VectorType>::Element: VectorTypeFamily,
{
    type Output = <<Source::Output as VectorType>::Element as VectorTypeFamily>::Vector4;

    fn emit(&self, ctx: &mut InstructionEmitter) -> Id {
        let src = self.0.emit(ctx);

        if <Self::Output as Type>::id() == <Source::Output as Type>::id()
            && [E::INDEX, E2::INDEX, E3::INDEX, E4::INDEX] == [0, 1, 2, 3]
        {
            // identical, no operation emitted
            return src;
        }

        let rty = ctx.type_id::<Self::Output>();
        ctx.pure_result_instruction(PureResultInstruction::OpVectorShuffle {
            result_type: rty,
            vector1: src,
            vector2: src,
            components: vec![E::INDEX, E2::INDEX, E3::INDEX, E4::INDEX],
        })
    }
}

#[derive(Debug)]
pub struct InstructionEmitter {
    head_slots: Vec<Instruction>,
    head_id_top: Id,
    types: HashMap<TypeId, Id>,
    constants: HashMap<Constant, Id>,
    variable_slots: Vec<Instruction>,
    variable_id_top: Id,
    descriptor_variables: HashMap<DescriptorRefKey, Id>,
    builtin_variables: HashMap<Builtin, Id>,
    entrypoint_interface_var_ids: Vec<Id>,
    main_instructions: Vec<Instruction>,
    main_instruction_ret_id_top: Id,
    load_cache: HashMap<(Id, Id), Id>,
    pure_result_cache: HashMap<PureResultInstruction, Id>,
}
impl InstructionEmitter {
    pub fn new() -> Self {
        Self {
            head_slots: Vec::new(),
            head_id_top: 0,
            types: HashMap::new(),
            constants: HashMap::new(),
            variable_slots: Vec::new(),
            variable_id_top: 0,
            descriptor_variables: HashMap::new(),
            builtin_variables: HashMap::new(),
            entrypoint_interface_var_ids: Vec::new(),
            main_instructions: Vec::new(),
            main_instruction_ret_id_top: 0,
            load_cache: HashMap::new(),
            pure_result_cache: HashMap::new(),
        }
    }

    pub fn type_id<T: Type>(&mut self) -> Id {
        self.type_id_of_val(T::id())
    }

    pub fn type_id_of_val(&mut self, id: TypeId) -> Id {
        if !self.types.contains_key(&id) {
            let tid = self.head_id_top;
            self.head_id_top += 1;
            let inst = id.make_instruction(tid, self);
            self.head_slots.push(inst);

            self.types.insert(id, tid);
            tid
        } else {
            self.types[&id]
        }
    }

    pub fn constant_id(&mut self, c: Constant) -> Id {
        if !self.constants.contains_key(&c) {
            let cid = self.head_id_top;
            self.head_id_top += 1;
            let inst = c.make_instruction(cid, self);
            self.head_slots.push(inst);

            self.constants.insert(c, cid);
            cid
        } else {
            self.constants[&c]
        }
    }

    pub fn descriptor_var_id<T: DescriptorType>(&mut self, d: &DescriptorRef<T>) -> Id {
        let dk = d.make_key();

        if !self.descriptor_variables.contains_key(&dk) {
            let did = self.variable_id_top;
            self.variable_id_top += 1;
            let ty = self.type_id::<UniformConstantPointer<T>>();
            self.variable_slots.push(Instruction::OpVariable {
                result_type: ty,
                result: did,
                storage_class: StorageClass::UniformConstant,
                initializer: None,
            });

            self.descriptor_variables.insert(dk, did);
            did
        } else {
            self.descriptor_variables[&dk]
        }
    }

    pub fn builtin_var_id<R: BuiltinRef>(&mut self) -> Id {
        let key = R::ID;

        if !self.builtin_variables.contains_key(&key) {
            let bid = self.variable_id_top;
            self.variable_id_top += 1;
            let ty = self.type_id::<Pointer<R::ValueType, R::StorageClass>>();
            self.variable_slots.push(Instruction::OpVariable {
                result_type: ty,
                result: bid,
                storage_class: <R::StorageClass>::VALUE,
                initializer: None,
            });

            if <R::StorageClass>::VALUE == StorageClass::Input
                || <R::StorageClass>::VALUE == StorageClass::Output
            {
                self.entrypoint_interface_var_ids.push(bid);
            }

            self.builtin_variables.insert(key, bid);
            bid
        } else {
            self.builtin_variables[&key]
        }
    }

    pub fn alloc_ret_id(&mut self) -> Id {
        let id = self.main_instruction_ret_id_top;
        self.main_instruction_ret_id_top += 1;

        id
    }

    pub fn load(&mut self, result_type: Id, pointer: Id) -> Id {
        let key = (result_type, pointer);

        if !self.load_cache.contains_key(&key) {
            let id = self.alloc_ret_id();
            let inst = Instruction::OpLoad {
                result_type,
                result: id,
                pointer,
                memory_operands: None,
            };
            self.add_main_instruction(inst);
            self.load_cache.insert(key, id);

            id
        } else {
            self.load_cache[&key]
        }
    }

    pub fn pure_result_instruction(&mut self, instruction: PureResultInstruction) -> Id {
        if !self.pure_result_cache.contains_key(&instruction) {
            let rid = self.alloc_ret_id();
            self.add_main_instruction(Instruction::PureResult(rid, instruction.clone()));
            self.pure_result_cache.insert(instruction, rid);

            rid
        } else {
            self.pure_result_cache[&instruction]
        }
    }

    pub fn add_main_instruction(&mut self, instruction: Instruction) {
        self.main_instructions.push(instruction);
    }

    /// (entrypoint_id, interface_ids, instructions, bound)
    pub fn serialize_instructions(mut self) -> (Id, Vec<Id>, Vec<Instruction>, Id) {
        let main_return_type_id = self.type_id::<Void>();
        let main_fn_type_id = self.type_id::<CallableType<(), Void>>();

        let head_base = 1;
        let variables_base = head_base + self.head_id_top;
        let entrypoint_id = variables_base + self.variable_id_top;
        let main_base = entrypoint_id + 2; // reserve OpLabel
        let bound = main_base + self.main_instruction_ret_id_top;

        // 3: reservations for OpFunction + OpFunctionEnd + OpReturn
        let mut serialized = Vec::with_capacity(
            self.builtin_variables.len()
                + self.descriptor_variables.len() * 2
                + self.head_slots.len()
                + self.variable_slots.len()
                + self.main_instructions.len()
                + 3,
        );
        serialized.extend(
            self.builtin_variables.into_iter().map(|(b, bid)| {
                Instruction::OpDecorate(bid + variables_base, Decoration::Builtin(b))
            }),
        );
        serialized.extend(self.descriptor_variables.into_iter().flat_map(|(k, bid)| {
            let mut instructions = vec![
                Instruction::OpDecorate(bid + variables_base, Decoration::DescriptorSet(k.set)),
                Instruction::OpDecorate(bid + variables_base, Decoration::Binding(k.bound)),
            ];
            if !k.decl.mutable {
                instructions.push(Instruction::OpDecorate(
                    bid + variables_base,
                    Decoration::NonWritable,
                ));
            }

            instructions
        }));
        serialized.extend(
            self.head_slots
                .into_iter()
                .map(|x| Self::translate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend(
            self.variable_slots
                .into_iter()
                .map(|x| Self::translate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend([
            Instruction::OpFunction {
                result_type: main_return_type_id + head_base,
                result: entrypoint_id,
                control: FunctionControl::NONE,
                r#type: main_fn_type_id + head_base,
            },
            Instruction::OpLabel(entrypoint_id + 1),
        ]);
        serialized.extend(
            self.main_instructions
                .into_iter()
                .map(|x| Self::translate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend([Instruction::OpReturn, Instruction::OpFunctionEnd]);

        (
            entrypoint_id,
            self.entrypoint_interface_var_ids
                .into_iter()
                .map(|x| x + variables_base)
                .collect(),
            serialized,
            bound,
        )
    }

    fn translate_ids(
        inst: Instruction,
        head_base: u32,
        variables_base: u32,
        main_base: u32,
    ) -> Instruction {
        match inst {
            Instruction::OpCapability(c) => Instruction::OpCapability(c),
            Instruction::OpMemoryModel(a, m) => Instruction::OpMemoryModel(a, m),
            Instruction::OpDecorate(a, b) => Instruction::OpDecorate(a, b),
            Instruction::OpEntryPoint {
                execution_model,
                func_id,
                name,
                interface,
            } => Instruction::OpEntryPoint {
                execution_model,
                func_id,
                name,
                interface,
            },
            Instruction::OpExecutionMode(a, b) => Instruction::OpExecutionMode(a, b),
            Instruction::OpTypeVoid(r) => Instruction::OpTypeVoid(r + head_base),
            Instruction::OpTypeBool(r) => Instruction::OpTypeBool(r + head_base),
            Instruction::OpTypeInt {
                result,
                bits,
                signed,
            } => Instruction::OpTypeInt {
                result: result + head_base,
                bits,
                signed,
            },
            Instruction::OpTypeFloat(r, b) => Instruction::OpTypeFloat(r + head_base, b),
            Instruction::OpTypeVector(r, e, n) => {
                Instruction::OpTypeVector(r + head_base, e + head_base, n)
            }
            Instruction::OpTypeMatrix(r, c, cn) => {
                Instruction::OpTypeMatrix(r + head_base, c + head_base, cn)
            }
            Instruction::OpTypeImage {
                result,
                sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampling_cap,
                format,
                qualifier,
            } => Instruction::OpTypeImage {
                result: result + head_base,
                sampled_type: sampled_type + head_base,
                dim,
                depth,
                arrayed,
                multisampled,
                sampling_cap,
                format,
                qualifier,
            },
            Instruction::OpTypeSampler(r) => Instruction::OpTypeSampler(r + head_base),
            Instruction::OpTypeSampledImage { result, image_type } => {
                Instruction::OpTypeSampledImage {
                    result: result + head_base,
                    image_type: image_type + head_base,
                }
            }
            Instruction::OpTypeArray {
                result,
                element_type,
                length_expr,
            } => Instruction::OpTypeArray {
                result: result + head_base,
                element_type: element_type + head_base,
                length_expr: length_expr + head_base,
            },
            Instruction::OpTypeRuntimeArray {
                result,
                element_type,
            } => Instruction::OpTypeRuntimeArray {
                result: result + head_base,
                element_type: element_type + head_base,
            },
            Instruction::OpTypePointer {
                result,
                storage_class,
                r#type,
            } => Instruction::OpTypePointer {
                result: result + head_base,
                storage_class,
                r#type: r#type + head_base,
            },
            Instruction::OpTypeFunction {
                result,
                return_type,
                mut parameter_types,
            } => {
                for v in parameter_types.iter_mut() {
                    *v += head_base;
                }

                Instruction::OpTypeFunction {
                    result: result + head_base,
                    return_type: return_type + head_base,
                    parameter_types: parameter_types,
                }
            }
            Instruction::OpConstantTrue {
                result_type,
                result,
            } => Instruction::OpConstantTrue {
                result_type: result_type + head_base,
                result: result + head_base,
            },
            Instruction::OpConstantFalse {
                result_type,
                result,
            } => Instruction::OpConstantFalse {
                result_type: result_type + head_base,
                result: result + head_base,
            },
            Instruction::OpConstant {
                result_type,
                result,
                value,
            } => Instruction::OpConstant {
                result_type: result_type + head_base,
                result: result + head_base,
                value,
            },
            Instruction::OpConstantComposite {
                result_type,
                result,
                consituents,
            } => Instruction::OpConstantComposite {
                result_type: result_type + head_base,
                result: result + head_base,
                consituents,
            },
            Instruction::OpConstantSampler {
                result_type,
                result,
                addressing,
                normalized,
                filter,
            } => Instruction::OpConstantSampler {
                result_type: result_type + head_base,
                result: result + head_base,
                addressing,
                normalized,
                filter,
            },
            Instruction::OpConstantNull {
                result_type,
                result,
            } => Instruction::OpConstantNull {
                result_type: result_type + head_base,
                result: result + head_base,
            },
            Instruction::PureResult(result, p) => Instruction::PureResult(
                result + main_base,
                match p {
                    PureResultInstruction::OpConvertFToS {
                        result_type,
                        float_value,
                    } => PureResultInstruction::OpConvertFToS {
                        result_type: result_type + head_base,
                        float_value: float_value + main_base,
                    },
                    PureResultInstruction::OpBitcast {
                        result_type,
                        operand,
                    } => PureResultInstruction::OpBitcast {
                        result_type: result_type + head_base,
                        operand: operand + main_base,
                    },
                    PureResultInstruction::OpVectorShuffle {
                        result_type,
                        vector1,
                        vector2,
                        components,
                    } => PureResultInstruction::OpVectorShuffle {
                        result_type: result_type + head_base,
                        vector1: vector1 + main_base,
                        vector2: vector2 + main_base,
                        components,
                    },
                    PureResultInstruction::OpCompositeExtract {
                        result_type,
                        composite,
                        indexes,
                    } => PureResultInstruction::OpCompositeExtract {
                        result_type: result_type + head_base,
                        composite: composite + main_base,
                        indexes,
                    },
                    PureResultInstruction::OpIAdd {
                        result_type,
                        operand1,
                        operand2,
                    } => PureResultInstruction::OpIAdd {
                        result_type: result_type + head_base,
                        operand1: operand1 + main_base,
                        operand2: operand2 + main_base,
                    },
                    PureResultInstruction::OpFAdd {
                        result_type,
                        operand1,
                        operand2,
                    } => PureResultInstruction::OpFAdd {
                        result_type: result_type + head_base,
                        operand1: operand1 + main_base,
                        operand2: operand2 + main_base,
                    },
                },
            ),
            Instruction::OpImageRead {
                result_type,
                result,
                image,
                coordinate,
                operands,
            } => Instruction::OpImageRead {
                result_type: result_type + head_base,
                result: result + main_base,
                image: image + main_base,
                coordinate: coordinate + main_base,
                operands,
            },
            Instruction::OpImageWrite {
                image,
                coordinate,
                texel,
                operands,
            } => Instruction::OpImageWrite {
                image: image + main_base,
                coordinate: coordinate + main_base,
                texel: texel + main_base,
                operands,
            },
            Instruction::OpReturn => Instruction::OpReturn,
            Instruction::OpReturnValue(v) => Instruction::OpReturnValue(v + main_base),
            Instruction::OpFunction {
                result_type,
                result,
                control,
                r#type,
            } => Instruction::OpFunction {
                result_type: result_type + head_base,
                result: result + main_base,
                control,
                r#type: r#type + head_base,
            },
            Instruction::OpFunctionEnd => Instruction::OpFunctionEnd,
            Instruction::OpLabel(r) => Instruction::OpLabel(r + main_base),
            Instruction::OpVariable {
                result_type,
                result,
                storage_class,
                initializer,
            } => Instruction::OpVariable {
                result_type: result_type + head_base,
                result: result + variables_base,
                storage_class,
                initializer,
            },
            Instruction::OpLoad {
                result_type,
                result,
                pointer,
                memory_operands,
            } => Instruction::OpLoad {
                result_type: result_type + head_base,
                result: result + main_base,
                pointer: pointer + variables_base,
                memory_operands,
            },
        }
    }
}

fn main() -> std::io::Result<()> {
    csh_accum2().export_module("accum2.spv")?;
    csh_accum3().export_module("accum3.spv")?;

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
