use crate::{
    instruction_emitter::{InstructionEmitter, RelativeId},
    spir::Builtin,
};

use super::{
    Float4, InputStorage, OutputStorage, ShaderExpression, ShaderRefExpression,
    ShaderStorableExpression, StorageClassMarker, Type, Uint3,
};

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
pub struct PositionRef;
impl BuiltinRef for PositionRef {
    const ID: Builtin = Builtin::Position;
    type StorageClass = OutputStorage;
    type ValueType = Float4;
}
impl ShaderExpression for PositionRef {
    type Output = Float4;

    fn emit(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        let vid = ctx.builtin_var_id::<Self>();
        let rty = ctx.type_id::<Self::Output>();

        ctx.load(rty, vid)
    }
}
impl ShaderRefExpression for PositionRef {
    type StorageClass = OutputStorage;

    fn emit_pointer(&self, ctx: &mut InstructionEmitter) -> RelativeId {
        ctx.builtin_var_id::<Self>()
    }
}
impl ShaderStorableExpression for PositionRef {}
