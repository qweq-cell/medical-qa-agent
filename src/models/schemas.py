"""
数据模型定义（Pydantic）
"""
from pydantic import BaseModel, Field
from typing import Optional


class MedicalEntity(BaseModel):
    """医疗实体"""
    name: str = Field(description="原文中的实体名")
    std_name: str = Field(description="标准化后的名称")
    category: str = Field(description="实体类型：疾病/症状/药品/检查/科室/食物/并发症/治疗方法")
    start: int = Field(0, description="在原文中的起始位置")
    end: int = Field(0, description="在原文中的结束位置")


class StructuredRecord(BaseModel):
    """结构化病历"""
    diseases: list[MedicalEntity] = Field(default_factory=list)
    symptoms: list[MedicalEntity] = Field(default_factory=list)
    drugs: list[MedicalEntity] = Field(default_factory=list)
    tests: list[MedicalEntity] = Field(default_factory=list)
    departments: list[MedicalEntity] = Field(default_factory=list)


class QCFinding(BaseModel):
    """质控发现"""
    type: str = Field(description="规则类型：gender_mismatch / drug_contraindication / dosage_abnormal / etc.")
    severity: str = Field(description="严重程度：severe / warning / info")
    description: str = Field(description="问题描述")
    suggestion: str = Field(default="", description="修改建议")
    entities: list[str] = Field(default_factory=list, description="涉及的实体名")


class QCSummary(BaseModel):
    """质控总结"""
    total_findings: int = 0
    severe_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    passed: bool = True


class QCReport(BaseModel):
    """质控报告"""
    original_text: str = Field(description="原始病历文本")
    entities: StructuredRecord = Field(default_factory=StructuredRecord)
    findings: list[QCFinding] = Field(default_factory=list)
    summary: QCSummary = Field(default_factory=QCSummary)
    llm_enabled: bool = Field(False, description="是否启用 LLM Agent 模式")
    llm_report: Optional[str] = Field(None, description="LLM 生成的自然语言质控报告")
    pipeline_mode: str = Field(
        "fixed",
        description="实际生效的管线：fixed（确定性）/ agent（ReAct 自主规划）/ agent_fallback（回退）",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "患者张伟，男，67岁，因口渴多饮就诊。诊断：2型糖尿病。",
                "entities": {
                    "diseases": [{"name": "2型糖尿病", "std_name": "2型糖尿病", "category": "疾病"}],
                    "symptoms": [{"name": "口渴多饮", "std_name": "口渴多饮", "category": "症状"}]
                },
                "findings": [],
                "summary": {"total_findings": 0, "severe_count": 0, "warning_count": 0, "info_count": 0, "passed": True}
            }
        }