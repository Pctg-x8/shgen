use std::collections::HashMap;

use crate::{
    dsl::{BuiltinRef, CallableType, Constant, Pointer, StorageClassMarker, Type, TypeId, Void},
    spir::{
        Builtin, Decoration, FunctionControl, Id, Instruction, PureResultInstruction, StorageClass,
    },
    DescriptorRef, DescriptorRefDefinition, DescriptorType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelativeId {
    Head(u32),
    Variable(u32),
    Main(u32),
}
impl RelativeId {
    pub fn relocate(self, head_base: u32, variables_base: u32, main_base: u32) -> Id {
        match self {
            Self::Head(x) => x + head_base,
            Self::Variable(x) => x + variables_base,
            Self::Main(x) => x + main_base,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorRefKey {
    set: u32,
    bound: u32,
    decl: DescriptorRefDefinition,
}

#[derive(Debug)]
pub struct InstructionEmitter {
    head_slots: Vec<Instruction<RelativeId>>,
    head_id_top: Id,
    type_decoration_slots: Vec<Instruction<RelativeId>>,
    types: HashMap<TypeId, Id>,
    constants: HashMap<Constant, Id>,
    variable_slots: Vec<Instruction<RelativeId>>,
    variable_id_top: Id,
    descriptor_variables: HashMap<DescriptorRefKey, Id>,
    builtin_variables: HashMap<Builtin, Id>,
    location_variables: HashMap<(StorageClass, u32), Id>,
    entrypoint_interface_var_ids: Vec<RelativeId>,
    main_instructions: Vec<Instruction<RelativeId>>,
    main_instruction_ret_id_top: Id,
    load_cache: HashMap<RelativeId, HashMap<RelativeId, RelativeId>>,
    pure_result_cache: HashMap<PureResultInstruction<RelativeId>, RelativeId>,
}
impl InstructionEmitter {
    pub fn new() -> Self {
        Self {
            head_slots: Vec::new(),
            head_id_top: 0,
            type_decoration_slots: Vec::new(),
            types: HashMap::new(),
            constants: HashMap::new(),
            variable_slots: Vec::new(),
            variable_id_top: 0,
            descriptor_variables: HashMap::new(),
            builtin_variables: HashMap::new(),
            location_variables: HashMap::new(),
            entrypoint_interface_var_ids: Vec::new(),
            main_instructions: Vec::new(),
            main_instruction_ret_id_top: 0,
            load_cache: HashMap::new(),
            pure_result_cache: HashMap::new(),
        }
    }

    pub fn type_id<T: Type>(&mut self) -> RelativeId {
        self.type_id_of_val(T::id())
    }

    pub fn type_id_of_val(&mut self, id: TypeId) -> RelativeId {
        RelativeId::Head(if !self.types.contains_key(&id) {
            let tid = self.head_id_top;
            self.head_id_top += 1;
            let inst = id.make_instruction(RelativeId::Head(tid), self);
            self.head_slots.push(inst);

            if let TypeId::Struct(ref members) = id {
                self.type_decoration_slots.push(Instruction::OpDecorate(
                    RelativeId::Head(tid),
                    Decoration::Block,
                ));
                self.type_decoration_slots
                    .extend(members.iter().enumerate().flat_map(|(n, m)| {
                        m.extra_decorations
                            .iter()
                            .cloned()
                            .chain([Decoration::Offset(m.offset)])
                            .map(move |d| {
                                Instruction::OpMemberDecorate(RelativeId::Head(tid), n as _, d)
                            })
                    }));
            }

            self.types.insert(id, tid);
            tid
        } else {
            self.types[&id]
        })
    }

    pub fn constant_id(&mut self, c: Constant) -> RelativeId {
        RelativeId::Head(if !self.constants.contains_key(&c) {
            let cid = self.head_id_top;
            self.head_id_top += 1;
            let inst = c.make_instruction(RelativeId::Head(cid), self);
            self.head_slots.push(inst);

            self.constants.insert(c, cid);
            cid
        } else {
            self.constants[&c]
        })
    }

    pub fn descriptor_var_id<T: DescriptorType>(&mut self, d: &DescriptorRef<T>) -> RelativeId {
        let dk = DescriptorRefKey {
            set: d.set,
            bound: d.bound,
            decl: T::DEF,
        };

        RelativeId::Variable(if !self.descriptor_variables.contains_key(&dk) {
            let did = self.variable_id_top;
            self.variable_id_top += 1;
            let ty = self.type_id::<Pointer<T, T::StorageClass>>();
            self.variable_slots.push(Instruction::OpVariable {
                result_type: ty,
                result: RelativeId::Variable(did),
                storage_class: <T::StorageClass>::VALUE,
                initializer: None,
            });

            self.descriptor_variables.insert(dk, did);
            did
        } else {
            self.descriptor_variables[&dk]
        })
    }

    pub fn builtin_var_id<R: BuiltinRef>(&mut self) -> RelativeId {
        let key = R::ID;

        RelativeId::Variable(if !self.builtin_variables.contains_key(&key) {
            let bid = self.variable_id_top;
            self.variable_id_top += 1;
            let ty = self.type_id::<Pointer<R::ValueType, R::StorageClass>>();
            self.variable_slots.push(Instruction::OpVariable {
                result_type: ty,
                result: RelativeId::Variable(bid),
                storage_class: <R::StorageClass>::VALUE,
                initializer: None,
            });

            if <R::StorageClass>::VALUE == StorageClass::Input
                || <R::StorageClass>::VALUE == StorageClass::Output
            {
                self.entrypoint_interface_var_ids
                    .push(RelativeId::Variable(bid));
            }

            self.builtin_variables.insert(key, bid);
            bid
        } else {
            self.builtin_variables[&key]
        })
    }

    pub fn location_var_id<T: Type, S: StorageClassMarker>(&mut self, location: u32) -> RelativeId {
        let key = (S::VALUE, location);

        if let Some(id) = self.location_variables.get(&key) {
            return RelativeId::Variable(*id);
        }

        let id = self.variable_id_top;
        self.variable_id_top += 1;
        let ty = self.type_id::<Pointer<T, S>>();
        self.variable_slots.push(Instruction::OpVariable {
            result_type: ty,
            result: RelativeId::Variable(id),
            storage_class: S::VALUE,
            initializer: None,
        });

        if S::VALUE == StorageClass::Input || S::VALUE == StorageClass::Output {
            self.entrypoint_interface_var_ids
                .push(RelativeId::Variable(id));
        }

        self.location_variables.insert(key, id);
        RelativeId::Variable(id)
    }

    pub fn alloc_ret_id(&mut self) -> RelativeId {
        let id = self.main_instruction_ret_id_top;
        self.main_instruction_ret_id_top += 1;

        RelativeId::Main(id)
    }

    pub fn load(&mut self, result_type: RelativeId, pointer: RelativeId) -> RelativeId {
        if let Some(c) = self.load_cache.get(&pointer) {
            if let Some(preloaded) = c.get(&result_type) {
                return *preloaded;
            }
        }

        let id = self.alloc_ret_id();
        let inst = Instruction::OpLoad {
            result_type,
            result: id,
            pointer,
            memory_operands: None,
        };
        self.add_main_instruction(inst);
        self.load_cache
            .entry(pointer)
            .or_insert_with(HashMap::new)
            .insert(result_type, id);

        id
    }

    pub fn store(&mut self, dest_ptr: RelativeId, value: RelativeId) {
        self.add_main_instruction(Instruction::OpStore {
            pointer: dest_ptr,
            object: value,
            memory_operands: None,
        });

        // TODO: AccessChainを考慮できてないので改修が必要
        // evict preloaded cache
        self.load_cache.remove(&dest_ptr);
    }

    pub fn pure_result_instruction(
        &mut self,
        instruction: PureResultInstruction<RelativeId>,
    ) -> RelativeId {
        if !self.pure_result_cache.contains_key(&instruction) {
            let rid = self.alloc_ret_id();
            self.add_main_instruction(Instruction::PureResult(rid, instruction.clone()));
            self.pure_result_cache.insert(instruction, rid);

            rid
        } else {
            self.pure_result_cache[&instruction]
        }
    }

    pub fn add_main_instruction(&mut self, instruction: Instruction<RelativeId>) {
        self.main_instructions.push(instruction);
    }

    /// (entrypoint_id, interface_ids, instructions, bound)
    pub fn serialize_instructions(mut self) -> (Id, Vec<Id>, Vec<Instruction<Id>>, Id) {
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
                + self.location_variables.len()
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
        serialized.extend(self.location_variables.into_iter().map(|((_, l), lid)| {
            Instruction::OpDecorate(lid + variables_base, Decoration::Location(l))
        }));
        serialized.extend(
            self.type_decoration_slots
                .into_iter()
                .map(|x| Self::relocate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend(
            self.head_slots
                .into_iter()
                .map(|x| Self::relocate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend(
            self.variable_slots
                .into_iter()
                .map(|x| Self::relocate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend([
            Instruction::OpFunction {
                result_type: main_return_type_id.relocate(head_base, variables_base, main_base),
                result: entrypoint_id,
                control: FunctionControl::NONE,
                r#type: main_fn_type_id.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpLabel(entrypoint_id + 1),
        ]);
        serialized.extend(
            self.main_instructions
                .into_iter()
                .map(|x| Self::relocate_ids(x, head_base, variables_base, main_base)),
        );
        serialized.extend([Instruction::OpReturn, Instruction::OpFunctionEnd]);

        (
            entrypoint_id,
            self.entrypoint_interface_var_ids
                .into_iter()
                .map(|x| x.relocate(head_base, variables_base, main_base))
                .collect(),
            serialized,
            bound,
        )
    }

    fn relocate_ids(
        inst: Instruction<RelativeId>,
        head_base: u32,
        variables_base: u32,
        main_base: u32,
    ) -> Instruction<Id> {
        match inst {
            Instruction::OpCapability(c) => Instruction::OpCapability(c),
            Instruction::OpMemoryModel(a, m) => Instruction::OpMemoryModel(a, m),
            Instruction::OpDecorate(a, b) => {
                Instruction::OpDecorate(a.relocate(head_base, variables_base, main_base), b)
            }
            Instruction::OpMemberDecorate(s, m, d) => Instruction::OpMemberDecorate(
                s.relocate(head_base, variables_base, main_base),
                m,
                d,
            ),
            Instruction::OpEntryPoint {
                execution_model,
                func_id,
                name,
                interface,
            } => Instruction::OpEntryPoint {
                execution_model,
                func_id: func_id.relocate(head_base, variables_base, main_base),
                name,
                interface: interface
                    .into_iter()
                    .map(|x| x.relocate(head_base, variables_base, main_base))
                    .collect(),
            },
            Instruction::OpExecutionMode(a, b) => {
                Instruction::OpExecutionMode(a.relocate(head_base, variables_base, main_base), b)
            }
            Instruction::OpTypeVoid(r) => {
                Instruction::OpTypeVoid(r.relocate(head_base, variables_base, main_base))
            }
            Instruction::OpTypeBool(r) => {
                Instruction::OpTypeBool(r.relocate(head_base, variables_base, main_base))
            }
            Instruction::OpTypeInt {
                result,
                bits,
                signed,
            } => Instruction::OpTypeInt {
                result: result.relocate(head_base, variables_base, main_base),
                bits,
                signed,
            },
            Instruction::OpTypeFloat(r, b) => {
                Instruction::OpTypeFloat(r.relocate(head_base, variables_base, main_base), b)
            }
            Instruction::OpTypeVector(r, e, n) => Instruction::OpTypeVector(
                r.relocate(head_base, variables_base, main_base),
                e.relocate(head_base, variables_base, main_base),
                n,
            ),
            Instruction::OpTypeMatrix(r, c, cn) => Instruction::OpTypeMatrix(
                r.relocate(head_base, variables_base, main_base),
                c.relocate(head_base, variables_base, main_base),
                cn,
            ),
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
                result: result.relocate(head_base, variables_base, main_base),
                sampled_type: sampled_type.relocate(head_base, variables_base, main_base),
                dim,
                depth,
                arrayed,
                multisampled,
                sampling_cap,
                format,
                qualifier,
            },
            Instruction::OpTypeSampler(r) => {
                Instruction::OpTypeSampler(r.relocate(head_base, variables_base, main_base))
            }
            Instruction::OpTypeSampledImage { result, image_type } => {
                Instruction::OpTypeSampledImage {
                    result: result.relocate(head_base, variables_base, main_base),
                    image_type: image_type.relocate(head_base, variables_base, main_base),
                }
            }
            Instruction::OpTypeArray {
                result,
                element_type,
                length_expr,
            } => Instruction::OpTypeArray {
                result: result.relocate(head_base, variables_base, main_base),
                element_type: element_type.relocate(head_base, variables_base, main_base),
                length_expr: length_expr.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpTypeRuntimeArray {
                result,
                element_type,
            } => Instruction::OpTypeRuntimeArray {
                result: result.relocate(head_base, variables_base, main_base),
                element_type: element_type.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpTypePointer {
                result,
                storage_class,
                r#type,
            } => Instruction::OpTypePointer {
                result: result.relocate(head_base, variables_base, main_base),
                storage_class,
                r#type: r#type.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpTypeFunction {
                result,
                return_type,
                parameter_types,
            } => Instruction::OpTypeFunction {
                result: result.relocate(head_base, variables_base, main_base),
                return_type: return_type.relocate(head_base, variables_base, main_base),
                parameter_types: parameter_types
                    .into_iter()
                    .map(|x| x.relocate(head_base, variables_base, main_base))
                    .collect(),
            },
            Instruction::OpTypeStruct {
                result,
                member_types,
            } => Instruction::OpTypeStruct {
                result: result.relocate(head_base, variables_base, main_base),
                member_types: member_types
                    .into_iter()
                    .map(|x| x.relocate(head_base, variables_base, main_base))
                    .collect(),
            },
            Instruction::OpConstantTrue {
                result_type,
                result,
            } => Instruction::OpConstantTrue {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpConstantFalse {
                result_type,
                result,
            } => Instruction::OpConstantFalse {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpConstant {
                result_type,
                result,
                value,
            } => Instruction::OpConstant {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                value,
            },
            Instruction::OpConstantComposite {
                result_type,
                result,
                consituents,
            } => Instruction::OpConstantComposite {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                consituents: consituents
                    .into_iter()
                    .map(|x| x.relocate(head_base, variables_base, main_base))
                    .collect(),
            },
            Instruction::OpConstantSampler {
                result_type,
                result,
                addressing,
                normalized,
                filter,
            } => Instruction::OpConstantSampler {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                addressing,
                normalized,
                filter,
            },
            Instruction::OpConstantNull {
                result_type,
                result,
            } => Instruction::OpConstantNull {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
            },
            Instruction::PureResult(result, p) => Instruction::PureResult(
                result.relocate(head_base, variables_base, main_base),
                match p {
                    PureResultInstruction::OpConvertFToS {
                        result_type,
                        float_value,
                    } => PureResultInstruction::OpConvertFToS {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        float_value: float_value.relocate(head_base, variables_base, main_base),
                    },
                    PureResultInstruction::OpBitcast {
                        result_type,
                        operand,
                    } => PureResultInstruction::OpBitcast {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        operand: operand.relocate(head_base, variables_base, main_base),
                    },
                    PureResultInstruction::OpVectorShuffle {
                        result_type,
                        vector1,
                        vector2,
                        components,
                    } => PureResultInstruction::OpVectorShuffle {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        vector1: vector1.relocate(head_base, variables_base, main_base),
                        vector2: vector2.relocate(head_base, variables_base, main_base),
                        components,
                    },
                    PureResultInstruction::OpCompositeConstruct {
                        result_type,
                        constituents,
                    } => PureResultInstruction::OpCompositeConstruct {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        constituents: constituents
                            .into_iter()
                            .map(|x| x.relocate(head_base, variables_base, main_base))
                            .collect(),
                    },
                    PureResultInstruction::OpCompositeExtract {
                        result_type,
                        composite,
                        indexes,
                    } => PureResultInstruction::OpCompositeExtract {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        composite: composite.relocate(head_base, variables_base, main_base),
                        indexes,
                    },
                    PureResultInstruction::OpIAdd {
                        result_type,
                        operand1,
                        operand2,
                    } => PureResultInstruction::OpIAdd {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        operand1: operand1.relocate(head_base, variables_base, main_base),
                        operand2: operand2.relocate(head_base, variables_base, main_base),
                    },
                    PureResultInstruction::OpFAdd {
                        result_type,
                        operand1,
                        operand2,
                    } => PureResultInstruction::OpFAdd {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        operand1: operand1.relocate(head_base, variables_base, main_base),
                        operand2: operand2.relocate(head_base, variables_base, main_base),
                    },
                    PureResultInstruction::OpVectorTimesScalar {
                        result_type,
                        vector,
                        scalar,
                    } => PureResultInstruction::OpVectorTimesScalar {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        vector: vector.relocate(head_base, variables_base, main_base),
                        scalar: scalar.relocate(head_base, variables_base, main_base),
                    },
                    PureResultInstruction::OpMatrixTimesVector {
                        result_type,
                        matrix,
                        vector,
                    } => PureResultInstruction::OpMatrixTimesVector {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        matrix: matrix.relocate(head_base, variables_base, main_base),
                        vector: vector.relocate(head_base, variables_base, main_base),
                    },
                    PureResultInstruction::OpAccessChain {
                        result_type,
                        base,
                        indexes,
                    } => PureResultInstruction::OpAccessChain {
                        result_type: result_type.relocate(head_base, variables_base, main_base),
                        base: base.relocate(head_base, variables_base, main_base),
                        indexes: indexes
                            .into_iter()
                            .map(|x| x.relocate(head_base, variables_base, main_base))
                            .collect(),
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
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                image: image.relocate(head_base, variables_base, main_base),
                coordinate: coordinate.relocate(head_base, variables_base, main_base),
                operands,
            },
            Instruction::OpImageWrite {
                image,
                coordinate,
                texel,
                operands,
            } => Instruction::OpImageWrite {
                image: image.relocate(head_base, variables_base, main_base),
                coordinate: coordinate.relocate(head_base, variables_base, main_base),
                texel: texel.relocate(head_base, variables_base, main_base),
                operands,
            },
            Instruction::OpReturn => Instruction::OpReturn,
            Instruction::OpReturnValue(v) => {
                Instruction::OpReturnValue(v.relocate(head_base, variables_base, main_base))
            }
            Instruction::OpFunction {
                result_type,
                result,
                control,
                r#type,
            } => Instruction::OpFunction {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                control,
                r#type: r#type.relocate(head_base, variables_base, main_base),
            },
            Instruction::OpFunctionEnd => Instruction::OpFunctionEnd,
            Instruction::OpLabel(r) => {
                Instruction::OpLabel(r.relocate(head_base, variables_base, main_base))
            }
            Instruction::OpVariable {
                result_type,
                result,
                storage_class,
                initializer,
            } => Instruction::OpVariable {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                storage_class,
                initializer: initializer.map(|x| x.relocate(head_base, variables_base, main_base)),
            },
            Instruction::OpLoad {
                result_type,
                result,
                pointer,
                memory_operands,
            } => Instruction::OpLoad {
                result_type: result_type.relocate(head_base, variables_base, main_base),
                result: result.relocate(head_base, variables_base, main_base),
                pointer: pointer.relocate(head_base, variables_base, main_base),
                memory_operands,
            },
            Instruction::OpStore {
                pointer,
                object,
                memory_operands,
            } => Instruction::OpStore {
                pointer: pointer.relocate(head_base, variables_base, main_base),
                object: object.relocate(head_base, variables_base, main_base),
                memory_operands,
            },
        }
    }
}
