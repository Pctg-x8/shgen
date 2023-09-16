//! Standard Portable Intermediate Representation defs

pub type Id = u32;
pub type ResultId = Id;

#[repr(C)]
pub struct ModuleBinaryHeader {
    pub magic_number: u32,
    pub version_number: u32,
    pub generator_magic_number: u32,
    pub bound: u32,
    pub reserved: u32,
}
impl ModuleBinaryHeader {
    pub const fn new(
        major_version: u8,
        minor_version: u8,
        generator_magic_number: u32,
        bound: u32,
    ) -> Self {
        Self {
            magic_number: 0x07230203,
            version_number: (major_version as u32) << 16 | (minor_version as u32) << 8,
            generator_magic_number,
            bound,
            reserved: 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8; 4 * 5] {
        unsafe { core::mem::transmute(self) }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Capability {
    Matrix,
    Shader,
    Geometry,
    Tessellation,
    Address,
    Linkage,
    Kernel,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ExecutionModel {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    GLCompute,
    Kernel,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Decoration {
    RelaxedPrecision,
    SpecId(u32),
    Block,
    BufferBlock,
    RowMajor,
    ColMajor,
    ArrayStride(u32),
    MatrixStride(u32),
    GLSLShared,
    GLSLPacked,
    CPacked,
    Builtin(Builtin),
    NoPerspective,
    Flat,
    Patch,
    Centroid,
    Sample,
    Invariant,
    Restrict,
    Aliased,
    Volatile,
    Constant,
    Coherent,
    NonWritable,
    NonReadable,
    Uniform,
    UniformId(Id),
    SaturatedConversion,
    Stream(u32),
    Location(u32),
    Component(u32),
    Index(u32),
    Binding(u32),
    DescriptorSet(u32),
    Offset(u32),
    InputAttachmentIndex(u32),
}
impl Decoration {
    const fn word_count(&self) -> u16 {
        match self {
            Self::SpecId(_)
            | Self::ArrayStride(_)
            | Self::MatrixStride(_)
            | Self::Builtin(_)
            | Self::UniformId(_)
            | Self::Stream(_)
            | Self::Location(_)
            | Self::Component(_)
            | Self::Index(_)
            | Self::Binding(_)
            | Self::DescriptorSet(_)
            | Self::Offset(_)
            | Self::InputAttachmentIndex(_) => 2,
            _ => 1,
        }
    }

    pub fn encode(&self, sink: &mut Vec<u32>) {
        match self {
            Self::RelaxedPrecision => sink.push(0),
            Self::SpecId(id) => sink.extend([1, *id]),
            Self::Block => sink.push(2),
            Self::BufferBlock => sink.push(3),
            Self::RowMajor => sink.push(4),
            Self::ColMajor => sink.push(5),
            Self::ArrayStride(s) => sink.extend([6, *s]),
            Self::MatrixStride(s) => sink.extend([7, *s]),
            Self::GLSLShared => sink.push(8),
            Self::GLSLPacked => sink.push(9),
            Self::CPacked => sink.push(10),
            Self::Builtin(b) => sink.extend([11, *b as _]),
            Self::NoPerspective => sink.push(13),
            Self::Flat => sink.push(14),
            Self::Patch => sink.push(15),
            Self::Centroid => sink.push(16),
            Self::Sample => sink.push(17),
            Self::Invariant => sink.push(18),
            Self::Restrict => sink.push(19),
            Self::Aliased => sink.push(20),
            Self::Volatile => sink.push(21),
            Self::Constant => sink.push(22),
            Self::Coherent => sink.push(23),
            Self::NonWritable => sink.push(24),
            Self::NonReadable => sink.push(25),
            Self::Uniform => sink.push(26),
            Self::UniformId(id) => sink.extend([27, *id]),
            Self::SaturatedConversion => sink.push(28),
            Self::Stream(id) => sink.extend([29, *id]),
            Self::Location(id) => sink.extend([30, *id]),
            Self::Component(c) => sink.extend([31, *c]),
            Self::Index(x) => sink.extend([32, *x]),
            Self::Binding(b) => sink.extend([33, *b]),
            Self::DescriptorSet(s) => sink.extend([34, *s]),
            Self::Offset(o) => sink.extend([35, *o]),
            Self::InputAttachmentIndex(x) => sink.extend([43, *x]),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ExecutionMode {
    LocalSize(u32, u32, u32),
}
impl ExecutionMode {
    const fn word_count(&self) -> u16 {
        match self {
            Self::LocalSize(_, _, _) => 4,
        }
    }

    fn encode(&self, sink: &mut Vec<u32>) {
        match self {
            Self::LocalSize(x, y, z) => sink.extend([17, *x, *y, *z]),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AddressingModel {
    Logical,
    Physical32,
    Physical64,
    PhysicalStorageBuffer64 = 5348,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MemoryModel {
    Simple,
    GLSL450,
    OpenCL,
    Vulkan,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageDepthFlag {
    NonDepth,
    Depth,
    Undefined,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageSamplingCompatibilityFlag {
    KnownAtRuntime,
    Sampling,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageFormat {
    Unknown,
    Rgba32f,
    Rgba16f,
    Rgba8,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AccessQualifier {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum StorageClass {
    UniformConstant,
    Input,
    Uniform,
    Output,
    Workgroup,
    CrossWorkgroup,
    Private,
    Function,
    Generic,
    PushConstant,
    AtomicCounter,
    Image,
    StorageBuffer,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageDimension {
    One,
    Two,
    Three,
    Cube,
    Rect,
    Buffer,
    SubpassData,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SamplerAddressingMode {
    None,
    ClampToEdge,
    Clamp,
    Repeat,
    RepeatMirrored,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SamplerFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Builtin {
    GlobalInvocationId = 28,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct FunctionControl(pub u32);
impl FunctionControl {
    pub const NONE: Self = Self(0);
    pub const INLINE: Self = Self(0x01);
    pub const DONT_INLINE: Self = Self(0x02);
    pub const PURE: Self = Self(0x04);
    pub const CONST: Self = Self(0x08);
}
impl core::ops::BitOr for FunctionControl {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for FunctionControl {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct MemoryOperands(pub u32);
impl MemoryOperands {
    pub const VOLATILE: Self = Self(0x01);
    pub const ALIGNED: Self = Self(0x02);
    pub const NON_TEMPORAL: Self = Self(0x04);
    pub const MAKE_POINTER_AVAILABLE: Self = Self(0x08);
    pub const MAKE_POINTER_VISIBLE: Self = Self(0x10);
    pub const NON_PRIVATE_POINTER: Self = Self(0x20);
}
impl core::ops::BitOr for MemoryOperands {
    type Output = MemoryOperands;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}
impl core::ops::BitOrAssign for MemoryOperands {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Instruction {
    OpCapability(Capability),
    OpMemoryModel(AddressingModel, MemoryModel),
    OpDecorate(Id, Decoration),
    OpEntryPoint {
        execution_model: ExecutionModel,
        func_id: Id,
        name: String,
        interface: Vec<Id>,
    },
    OpExecutionMode(Id, ExecutionMode),
    OpTypeVoid(ResultId),
    OpTypeBool(ResultId),
    OpTypeInt {
        result: ResultId,
        bits: u32,
        signed: bool,
    },
    OpTypeFloat(ResultId, u32),
    OpTypeVector(ResultId, Id, u32),
    OpTypeMatrix(ResultId, Id, u32),
    OpTypeImage {
        result: ResultId,
        sampled_type: Id,
        dim: ImageDimension,
        depth: ImageDepthFlag,
        arrayed: bool,
        multisampled: bool,
        sampling_cap: ImageSamplingCompatibilityFlag,
        format: ImageFormat,
        qualifier: Option<AccessQualifier>,
    },
    OpTypeSampler(ResultId),
    OpTypeSampledImage {
        result: ResultId,
        image_type: Id,
    },
    OpTypeArray {
        result: ResultId,
        element_type: Id,
        length_expr: Id,
    },
    OpTypeRuntimeArray {
        result: ResultId,
        element_type: Id,
    },
    OpTypePointer {
        result: ResultId,
        storage_class: StorageClass,
        r#type: Id,
    },
    OpTypeFunction {
        result: ResultId,
        return_type: Id,
        parameter_types: Vec<Id>,
    },
    OpConstantTrue {
        result_type: Id,
        result: ResultId,
    },
    OpConstantFalse {
        result_type: Id,
        result: ResultId,
    },
    OpConstant {
        result_type: Id,
        result: ResultId,
        value: Vec<u32>,
    },
    OpConstantComposite {
        result_type: Id,
        result: ResultId,
        consituents: Vec<Id>,
    },
    OpConstantSampler {
        result_type: Id,
        result: ResultId,
        addressing: SamplerAddressingMode,
        normalized: bool,
        filter: SamplerFilterMode,
    },
    OpConstantNull {
        result_type: Id,
        result: ResultId,
    },
    OpConvertFToS {
        result_type: Id,
        result: ResultId,
        float_value: Id,
    },
    OpBitcast {
        result_type: Id,
        result: ResultId,
        operand: Id,
    },
    OpVectorShuffle {
        result_type: Id,
        result: ResultId,
        vector1: Id,
        vector2: Id,
        components: Vec<u32>,
    },
    OpCompositeExtract {
        result_type: Id,
        result: ResultId,
        composite: Id,
        indexes: Vec<u32>,
    },
    OpIAdd {
        result_type: Id,
        result: ResultId,
        operand1: Id,
        operand2: Id,
    },
    OpFAdd {
        result_type: Id,
        result: ResultId,
        operand1: Id,
        operand2: Id,
    },
    OpImageRead {
        result_type: Id,
        result: ResultId,
        image: Id,
        coordinate: Id,
        operands: Option<u32>, // TODO: operands
    },
    OpImageWrite {
        image: Id,
        coordinate: Id,
        texel: Id,
        operands: Option<u32>, // TODO: operands
    },
    OpReturn,
    OpReturnValue(Id),
    OpFunction {
        result_type: Id,
        result: ResultId,
        control: FunctionControl,
        r#type: Id,
    },
    OpFunctionEnd,
    OpLabel(ResultId),
    OpVariable {
        result_type: Id,
        result: ResultId,
        storage_class: StorageClass,
        initializer: Option<Id>,
    },
    OpLoad {
        result_type: Id,
        result: ResultId,
        pointer: Id,
        memory_operands: Option<MemoryOperands>,
    },
}
impl Instruction {
    pub fn encode(self, sink: &mut Vec<u32>) {
        match self {
            Self::OpCapability(cap) => sink.extend([instruction_word(2, 17), cap as _]),
            Self::OpMemoryModel(a, m) => sink.extend([instruction_word(3, 14), a as _, m as _]),
            Self::OpDecorate(id, deco) => {
                sink.extend([instruction_word(2 + deco.word_count(), 71), id]);
                deco.encode(sink);
            }
            Self::OpEntryPoint {
                execution_model,
                func_id,
                name,
                interface,
            } => {
                let mut name_bytes = name.into_bytes();
                // nul-terminated
                name_bytes.push(0);
                // pad for 4 bytes
                name_bytes.resize((name_bytes.len() + 3) & !3, 0);
                sink.extend([
                    instruction_word(3 + name_bytes.len() as u16 / 4 + interface.len() as u16, 15),
                    execution_model as _,
                    func_id as _,
                ]);
                sink.extend(
                    name_bytes
                        .chunks_exact(4)
                        .map(|xs| u32::from_le_bytes(unsafe { xs.try_into().unwrap_unchecked() })),
                );
                sink.extend(interface);
            }
            Self::OpExecutionMode(id, e) => {
                sink.extend([instruction_word(2 + e.word_count(), 16), id]);
                e.encode(sink);
            }
            Self::OpTypeVoid(r) => sink.extend([instruction_word(2, 19), r]),
            Self::OpTypeBool(r) => sink.extend([instruction_word(2, 20), r]),
            Self::OpTypeInt {
                result,
                bits,
                signed,
            } => sink.extend([
                instruction_word(4, 21),
                result,
                bits,
                if signed { 1 } else { 0 },
            ]),
            Self::OpTypeFloat(r, bits) => sink.extend([instruction_word(3, 22), r, bits]),
            Self::OpTypeVector(r, e, c) => sink.extend([instruction_word(4, 23), r, e, c]),
            Self::OpTypeMatrix(r, c, cn) => sink.extend([instruction_word(4, 24), r, c, cn]),
            Self::OpTypeImage {
                result,
                sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampling_cap,
                format,
                qualifier,
            } => {
                sink.extend([
                    instruction_word(9 + if qualifier.is_some() { 1 } else { 0 }, 25),
                    result,
                    sampled_type,
                    dim as _,
                    depth as _,
                    if arrayed { 1 } else { 0 },
                    if multisampled { 1 } else { 0 },
                    sampling_cap as _,
                    format as _,
                ]);
                if let Some(q) = qualifier {
                    sink.push(q as _);
                }
            }
            Self::OpTypeSampler(r) => sink.extend([instruction_word(2, 26), r]),
            Self::OpTypeSampledImage { result, image_type } => {
                sink.extend([instruction_word(3, 27), result, image_type])
            }
            Self::OpTypeArray {
                result,
                element_type,
                length_expr,
            } => sink.extend([instruction_word(4, 28), result, element_type, length_expr]),
            Self::OpTypeRuntimeArray {
                result,
                element_type,
            } => sink.extend([instruction_word(3, 29), result, element_type]),
            Self::OpTypePointer {
                result,
                storage_class,
                r#type,
            } => sink.extend([instruction_word(4, 32), result, storage_class as _, r#type]),
            Self::OpTypeFunction {
                result,
                return_type,
                parameter_types,
            } => {
                sink.extend([
                    instruction_word(3 + parameter_types.len() as u16, 33),
                    result,
                    return_type,
                ]);
                sink.extend(parameter_types);
            }
            Self::OpConstantTrue {
                result_type,
                result,
            } => sink.extend([instruction_word(3, 41), result_type, result]),
            Self::OpConstantFalse {
                result_type,
                result,
            } => sink.extend([instruction_word(3, 42), result_type, result]),
            Self::OpConstant {
                result_type,
                result,
                value,
            } => {
                sink.extend([
                    instruction_word(3 + value.len() as u16, 43),
                    result_type,
                    result,
                ]);
                sink.extend(value);
            }
            Self::OpConstantComposite {
                result_type,
                result,
                consituents,
            } => {
                sink.extend([
                    instruction_word(3 + consituents.len() as u16, 44),
                    result_type,
                    result,
                ]);
                sink.extend(consituents);
            }
            Self::OpConstantSampler {
                result_type,
                result,
                addressing,
                normalized,
                filter,
            } => sink.extend([
                instruction_word(6, 45),
                result_type,
                result,
                addressing as _,
                if normalized { 1 } else { 0 },
                filter as _,
            ]),
            Self::OpConstantNull {
                result_type,
                result,
            } => sink.extend([instruction_word(3, 46), result_type, result]),
            Self::OpConvertFToS {
                result_type,
                result,
                float_value,
            } => sink.extend([instruction_word(4, 110), result_type, result, float_value]),
            Self::OpBitcast {
                result_type,
                result,
                operand,
            } => sink.extend([instruction_word(4, 124), result_type, result, operand]),
            Self::OpVectorShuffle {
                result_type,
                result,
                vector1,
                vector2,
                components,
            } => {
                sink.extend([
                    instruction_word(5 + components.len() as u16, 79),
                    result_type,
                    result,
                    vector1,
                    vector2,
                ]);
                sink.extend(components);
            }
            Self::OpCompositeExtract {
                result_type,
                result,
                composite,
                indexes,
            } => {
                sink.extend([
                    instruction_word(4 + indexes.len() as u16, 81),
                    result_type,
                    result,
                    composite,
                ]);
                sink.extend(indexes);
            }
            Self::OpIAdd {
                result_type,
                result,
                operand1,
                operand2,
            } => sink.extend([
                instruction_word(5, 128),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            Self::OpFAdd {
                result_type,
                result,
                operand1,
                operand2,
            } => sink.extend([
                instruction_word(5, 129),
                result_type,
                result,
                operand1,
                operand2,
            ]),
            Self::OpImageRead {
                result_type,
                result,
                image,
                coordinate,
                operands,
            } => {
                sink.extend([
                    instruction_word(5 + if operands.is_some() { 1 } else { 0 }, 98),
                    result_type,
                    result,
                    image,
                    coordinate,
                ]);
                if let Some(x) = operands {
                    sink.push(x);
                }
            }
            Self::OpImageWrite {
                image,
                coordinate,
                texel,
                operands,
            } => {
                sink.extend([
                    instruction_word(4 + if operands.is_some() { 1 } else { 0 }, 99),
                    image,
                    coordinate,
                    texel,
                ]);
                if let Some(x) = operands {
                    sink.push(x);
                }
            }
            Self::OpReturn => sink.push(instruction_word(1, 253)),
            Self::OpReturnValue(r) => sink.extend([instruction_word(2, 254), r]),
            Self::OpFunction {
                result_type,
                result,
                control,
                r#type,
            } => sink.extend([
                instruction_word(5, 54),
                result_type,
                result,
                control.0,
                r#type,
            ]),
            Self::OpFunctionEnd => sink.push(instruction_word(1, 56)),
            Self::OpLabel(r) => sink.extend([instruction_word(2, 248), r]),
            Self::OpVariable {
                result_type,
                result,
                storage_class,
                initializer,
            } => {
                sink.extend([
                    instruction_word(4 + if initializer.is_some() { 1 } else { 0 }, 59),
                    result_type,
                    result,
                    storage_class as _,
                ]);
                if let Some(x) = initializer {
                    sink.push(x);
                }
            }
            Self::OpLoad {
                result_type,
                result,
                pointer,
                memory_operands,
            } => {
                sink.extend([
                    instruction_word(4 + if memory_operands.is_some() { 1 } else { 0 }, 61),
                    result_type,
                    result,
                    pointer,
                ]);
                if let Some(x) = memory_operands {
                    sink.push(x.0);
                }
            }
        }
    }
}

const fn instruction_word(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}
