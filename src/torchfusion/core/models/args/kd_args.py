from dataclasses import dataclass, field


@dataclass
class KnowledgeDistillationArguments:
    model_mode: str = field(
        default="distillation",
        metadata={
            "help": "The step for knowledge distillation, Options: teacher, student, distillation"
        },
    )
    distillation_type: str = field(
        default="kd", metadata={"help": "The training mode. Options: kd, kd+ce, ce"}
    )
    variational_information_distillation_factor: float = field(
        default=0.1,
        metadata={
            "help": "The scaling factor for variational information distillation"
        },
    )
    knowledge_distillation_factor: float = field(
        default=1.0, metadata={"help": "The scaling factor for knowledge distillation"}
    )
    temperature: float = field(
        default=2.0, metadata={"help": "The temperature for distillation"}
    )
    teacher_model_constructor_args: dict = field(
        default_factory=dict,
        metadata={"help": "The arguments for the model constructor."},
    )
