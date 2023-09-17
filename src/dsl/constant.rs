use crate::{
    instruction_emitter::{InstructionEmitter, RelativeId},
    spir::Instruction,
    SafeFloat,
};

use super::{
    Bool, Float, Float2, Float3, Float4, Int, Int2, Int3, Int4, TypeId, Uint, Uint2, Uint3, Uint4,
};

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
