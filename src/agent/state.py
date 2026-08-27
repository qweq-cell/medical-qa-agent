"""
Agent 状态管理
"""
from dataclasses import dataclass, field
from src.models.schemas import StructuredRecord, QCFinding


@dataclass
class AgentState:
    """Agent 运行状态"""
    original_text: str = ""
    gender: str = ""  # male / female / unknown
    entities: StructuredRecord = field(default_factory=StructuredRecord)
    kg_results: dict = field(default_factory=dict)
    findings: list[QCFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text_length": len(self.original_text),
            "gender": self.gender,
            "entity_count": {
                "diseases": len(self.entities.diseases),
                "symptoms": len(self.entities.symptoms),
                "drugs": len(self.entities.drugs),
                "tests": len(self.entities.tests),
            },
            "finding_count": len(self.findings),
            "error_count": len(self.errors),
        }