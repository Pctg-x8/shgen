mod types;
use std::{rc::Rc, sync::Arc};

use crate::{
    instruction_emitter::{InstructionEmitter, RelativeId},
    spir::{
        Builtin, ImageDepthFlag, ImageDimension, ImageFormat, ImageSamplingCompatibilityFlag,
        Instruction, PureResultInstruction,
    },
    Descriptor, DescriptorRefDefinition, SafeFloat,
};

pub use self::types::*;

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
    pub fn make_instruction(
        &self,
        result: RelativeId,
        ctx: &mut InstructionEmitter,
    ) -> Instruction<RelativeId> {
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
    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId;

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

    fn mul<Right>(self, right: Right) -> Mul<Self, Right>
    where
        Self: Sized,
        Right: ShaderExpression,
        Self::Output: TypeMulRelation<Right::Output>,
    {
        Mul(self, right)
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        T::emit(&self, ctx)
    }
}
impl<T: ShaderExpression> ShaderExpression for Arc<T> {
    type Output = T::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        T::emit(&self, ctx)
    }
}
impl ShaderExpression for SafeFloat {
    type Output = Float;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        ctx.constant_id(Constant::Float(*self))
    }
}

pub trait ShaderRefExpression: ShaderExpression {
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId;

    fn store<V: ShaderExpression<Output = Self::Output>>(self, value: V) -> Store<Self, V>
    where
        Self: Sized,
    {
        Store(self, value)
    }
}
impl<T: ShaderRefExpression> ShaderRefExpression for Rc<T> {
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        T::emit_pointer(&self, ctx)
    }
}
impl<T: ShaderRefExpression> ShaderRefExpression for Arc<T> {
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        T::emit_pointer(&self, ctx)
    }
}

pub trait ShaderExpressionList: core::fmt::Debug {
    type Type: TypeList;

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<RelativeId>;
}
impl<T> ShaderExpressionList for (T,)
where
    T: ShaderExpression,
{
    type Type = (T::Output,);

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<RelativeId> {
        vec![self.0.emit(ctx)]
    }
}
impl<A, B> ShaderExpressionList for (A, B)
where
    A: ShaderExpression,
    B: ShaderExpression,
{
    type Type = (A::Output, B::Output);

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<RelativeId> {
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

    fn emit_all(&self, ctx: &mut InstructionEmitter) -> Vec<RelativeId> {
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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

#[derive(Clone, Copy, Debug)]
pub struct DescriptorRef<T: DescriptorType> {
    pub set: u32,
    pub bound: u32,
    pub _ph: core::marker::PhantomData<T>,
}
impl<T: DescriptorType> ShaderExpression for DescriptorRef<T> {
    type Output = T;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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
pub struct LocationRef<T: Type, S: StorageClassMarker>(
    pub u32,
    pub core::marker::PhantomData<(T, S)>,
);
impl<T: Type, S: StorageClassMarker> ShaderExpression for LocationRef<T, S> {
    type Output = T;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let vid = ctx.location_var_id::<T, S>(self.0);
        let tid = ctx.type_id::<T>();

        ctx.load(tid, vid)
    }
}
impl<T: Type> ShaderRefExpression for LocationRef<T, OutputStorage> {
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        ctx.location_var_id::<T, OutputStorage>(self.0)
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let callable = self.0.emit(ctx);
        let args = self.1.emit_all(ctx);
        let result_ty = ctx.type_id::<R>();

        // TODO: emit instruction
        RelativeId::Main(0)
    }
}

#[derive(Debug, Clone)]
pub struct Add<Left: ShaderExpression, Right: ShaderExpression>(Left, Right);
impl<Left: ShaderExpression, Right: ShaderExpression> ShaderExpression for Add<Left, Right>
where
    Left::Output: TypeAddRelation<Right::Output>,
{
    type Output = <Left::Output as TypeAddRelation<Right::Output>>::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let lhs = self.0.emit(ctx);
        let rhs = self.1.emit(ctx);

        <Left::Output as TypeAddRelation<Right::Output>>::emit(lhs, rhs, ctx)
    }
}

#[derive(Debug, Clone)]
pub struct Mul<Left: ShaderExpression, Right: ShaderExpression>(Left, Right);
impl<Left: ShaderExpression, Right: ShaderExpression> ShaderExpression for Mul<Left, Right>
where
    Left::Output: TypeMulRelation<Right::Output>,
{
    type Output = <Left::Output as TypeMulRelation<Right::Output>>::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let lhs = self.0.emit(ctx);
        let rhs = self.1.emit(ctx);

        <Left::Output as TypeMulRelation<Right::Output>>::emit(lhs, rhs, ctx)
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let src = self.0.emit(ctx);

        // Note: never identical (vector-type <> scalar-type)

        let rty = ctx.type_id::<Self::Output>();
        ctx.pure_result_instruction(PureResultInstruction::OpCompositeExtract {
            result_type: rty,
            composite: src,
            indexes: vec![E::INDEX],
        })
    }
}
impl<Source: ShaderRefExpression, E: VectorElementOf<Source::Output>> ShaderRefExpression
    for VectorSwizzle1<Source, E>
where
    Source::Output: VectorType,
{
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let src = self.0.emit_pointer(ctx);
        let e1 = ctx.constant_id(Constant::Uint(E::INDEX));
        let rty = ctx.type_id::<Self::Output>();

        ctx.pure_result_instruction(PureResultInstruction::OpAccessChain {
            result_type: rty,
            base: src,
            indexes: vec![e1],
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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
impl<
        Source: ShaderRefExpression,
        E: VectorElementOf<Source::Output>,
        E2: VectorElementOf<Source::Output>,
    > ShaderRefExpression for VectorSwizzle2<Source, E, E2>
where
    Source::Output: VectorType,
    <Source::Output as VectorType>::Element: VectorTypeFamily,
{
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let src = self.0.emit_pointer(ctx);
        let e1 = ctx.constant_id(Constant::Uint(E::INDEX));
        let e2 = ctx.constant_id(Constant::Uint(E2::INDEX));
        let rty = ctx.type_id::<Self::Output>();

        ctx.pure_result_instruction(PureResultInstruction::OpAccessChain {
            result_type: rty,
            base: src,
            indexes: vec![e1, e2],
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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
impl<
        Source: ShaderRefExpression,
        E: VectorElementOf<Source::Output>,
        E2: VectorElementOf<Source::Output>,
        E3: VectorElementOf<Source::Output>,
    > ShaderRefExpression for VectorSwizzle3<Source, E, E2, E3>
where
    Source::Output: VectorType,
    <Source::Output as VectorType>::Element: VectorTypeFamily,
{
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let src = self.0.emit_pointer(ctx);
        let e1 = ctx.constant_id(Constant::Uint(E::INDEX));
        let e2 = ctx.constant_id(Constant::Uint(E2::INDEX));
        let e3 = ctx.constant_id(Constant::Uint(E3::INDEX));
        let rty = ctx.type_id::<Self::Output>();

        ctx.pure_result_instruction(PureResultInstruction::OpAccessChain {
            result_type: rty,
            base: src,
            indexes: vec![e1, e2, e3],
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

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
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
impl<
        Source: ShaderRefExpression,
        E: VectorElementOf<Source::Output>,
        E2: VectorElementOf<Source::Output>,
        E3: VectorElementOf<Source::Output>,
        E4: VectorElementOf<Source::Output>,
    > ShaderRefExpression for VectorSwizzle4<Source, E, E2, E3, E4>
where
    Source::Output: VectorType,
    <Source::Output as VectorType>::Element: VectorTypeFamily,
{
    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let src = self.0.emit_pointer(ctx);
        let e1 = ctx.constant_id(Constant::Uint(E::INDEX));
        let e2 = ctx.constant_id(Constant::Uint(E2::INDEX));
        let e3 = ctx.constant_id(Constant::Uint(E3::INDEX));
        let e4 = ctx.constant_id(Constant::Uint(E4::INDEX));
        let rty = ctx.type_id::<Self::Output>();

        ctx.pure_result_instruction(PureResultInstruction::OpAccessChain {
            result_type: rty,
            base: src,
            indexes: vec![e1, e2, e3, e4],
        })
    }
}

#[derive(Debug, Clone)]
pub struct Composite2<A: ShaderExpression, B: ShaderExpression>(pub A, pub B);
impl<A: ShaderExpression, B: ShaderExpression> ShaderExpression for Composite2<A, B>
where
    A::Output: TypeCompositionRelation2<B::Output>,
{
    type Output = <A::Output as TypeCompositionRelation2<B::Output>>::Output;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let a0 = self.0.emit(ctx);
        let a1 = self.1.emit(ctx);
        let rty = ctx.type_id::<Self::Output>();

        ctx.pure_result_instruction(PureResultInstruction::OpCompositeConstruct {
            result_type: rty,
            constituents: vec![a0, a1],
        })
    }
}

#[derive(Debug, Clone)]
pub struct Store<Dest: ShaderRefExpression, Value: ShaderExpression>(Dest, Value);
impl<
        Dest: ShaderRefExpression,
        Value: ShaderExpression<Output = <Dest as ShaderExpression>::Output>,
    > ShaderAction for Store<Dest, Value>
{
    fn emit(&self, ctx: &mut InstructionEmitter) {
        let dst = self.0.emit_pointer(ctx);
        let value = self.1.emit(ctx);

        ctx.store(dst, value);
    }
}
