use crate::{
    instruction_emitter::{InstructionEmitter, RelativeId},
    spir::{
        AccessQualifier, ImageDepthFlag, ImageDimension, ImageFormat,
        ImageSamplingCompatibilityFlag, Instruction, PureResultInstruction, StorageClass,
    },
};

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
    pub fn make_instruction(
        &self,
        result: RelativeId,
        ctx: &mut InstructionEmitter,
    ) -> Instruction<RelativeId> {
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

pub trait Type: core::fmt::Debug {
    fn id() -> TypeId;
}
pub trait TypeAddRelation<Right: Type> {
    type Output: Type;

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId;
}
pub trait TypeMulRelation<Right: Type> {
    type Output: Type;

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId;
}
pub trait TypeCompositionRelation2<Other: Type> {
    type Output: Type;
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

pub fn image_type_id<T: ImageType>() -> TypeId {
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

#[derive(Debug, Clone, Copy)]
pub struct OutputStorage;
impl StorageClassMarker for OutputStorage {
    const VALUE: StorageClass = StorageClass::Output;
}

#[derive(Clone, Copy, Debug)]
pub struct Pointer<T: Type, C: StorageClassMarker>(core::marker::PhantomData<(T, C)>);
impl<T: Type, C: StorageClassMarker> Type for Pointer<T, C> {
    fn id() -> TypeId {
        TypeId::Pointer(Box::new(T::id()), C::VALUE)
    }
}

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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Float> for Float {
    type Output = Float2;
}
impl TypeCompositionRelation2<Float2> for Float {
    type Output = Float3;
}
impl TypeCompositionRelation2<Float3> for Float {
    type Output = Float4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Uint>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Uint> for Uint {
    type Output = Uint2;
}
impl TypeCompositionRelation2<Uint2> for Uint {
    type Output = Uint3;
}
impl TypeCompositionRelation2<Uint3> for Uint {
    type Output = Uint4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Int>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Int> for Int {
    type Output = Int2;
}
impl TypeCompositionRelation2<Int2> for Int {
    type Output = Int3;
}
impl TypeCompositionRelation2<Int3> for Int {
    type Output = Int4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float2>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeMulRelation<Float> for Float2 {
    type Output = Float2;

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float2>();

        ctx.pure_result_instruction(PureResultInstruction::OpVectorTimesScalar {
            result_type: rty,
            vector: left,
            scalar: right,
        })
    }
}
impl TypeCompositionRelation2<Float> for Float2 {
    type Output = Float3;
}
impl TypeCompositionRelation2<Float2> for Float2 {
    type Output = Float4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float3>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeMulRelation<Float> for Float3 {
    type Output = Float3;

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float3>();

        ctx.pure_result_instruction(PureResultInstruction::OpVectorTimesScalar {
            result_type: rty,
            vector: left,
            scalar: right,
        })
    }
}
impl TypeCompositionRelation2<Float> for Float3 {
    type Output = Float4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float4>();

        ctx.pure_result_instruction(PureResultInstruction::OpFAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeMulRelation<Float> for Float4 {
    type Output = Float4;

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Float4>();

        ctx.pure_result_instruction(PureResultInstruction::OpVectorTimesScalar {
            result_type: rty,
            vector: left,
            scalar: right,
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Int2>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Int> for Int2 {
    type Output = Int3;
}
impl TypeCompositionRelation2<Int2> for Int2 {
    type Output = Int4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Int3>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Int> for Int3 {
    type Output = Int4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Uint2>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Uint> for Uint2 {
    type Output = Uint3;
}
impl TypeCompositionRelation2<Uint2> for Uint2 {
    type Output = Uint4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Uint3>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
    }
}
impl TypeCompositionRelation2<Uint> for Uint3 {
    type Output = Uint4;
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

    fn emit(left: RelativeId, right: RelativeId, ctx: &mut InstructionEmitter) -> RelativeId {
        let rty = ctx.type_id::<Uint4>();

        ctx.pure_result_instruction(PureResultInstruction::OpIAdd {
            result_type: rty,
            operand1: left,
            operand2: right,
        })
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
