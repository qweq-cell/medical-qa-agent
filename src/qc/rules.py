"""
质控规则引擎
"""
import re
from src.models.schemas import QCFinding, StructuredRecord
from src.config import (
    FEMALE_ONLY_TESTS, MALE_ONLY_TESTS,
    FEMALE_ONLY_DRUGS, MALE_ONLY_DRUGS,
)


class QCRules:
    """质控规则集合"""

    @staticmethod
    def check_gender_mismatch(text: str, entities: StructuredRecord) -> list[QCFinding]:
        """
        性别矛盾检测：
        - 男性 + 妇科检查/药品/诊断
        - 女性 + 男性检查/药品/诊断
        """
        findings = []

        # 提取性别
        gender = None
        if re.search(r'(男|男性|先生)', text):
            gender = "male"
        elif re.search(r'(女|女性|女士)', text):
            gender = "female"

        if not gender:
            return findings

        # 检查检查项目
        for test in entities.tests:
            test_name = test.std_name
            if gender == "male" and any(kw in test_name for kw in FEMALE_ONLY_TESTS):
                findings.append(QCFinding(
                    type="gender_mismatch",
                    severity="severe",
                    description=f"男性患者涉及女性特有检查：{test_name}",
                    suggestion="请确认检查项目是否与患者性别匹配",
                    entities=[test_name]
                ))
            elif gender == "female" and any(kw in test_name for kw in MALE_ONLY_TESTS):
                findings.append(QCFinding(
                    type="gender_mismatch",
                    severity="severe",
                    description=f"女性患者涉及男性特有检查：{test_name}",
                    suggestion="请确认检查项目是否与患者性别匹配",
                    entities=[test_name]
                ))

        # 检查药品
        for drug in entities.drugs:
            drug_name = drug.std_name
            if gender == "male" and any(kw in drug_name for kw in FEMALE_ONLY_DRUGS):
                findings.append(QCFinding(
                    type="gender_mismatch",
                    severity="severe",
                    description=f"男性患者使用女性专用药品：{drug_name}",
                    suggestion="请确认用药是否与患者性别匹配",
                    entities=[drug_name]
                ))
            elif gender == "female" and any(kw in drug_name for kw in MALE_ONLY_DRUGS):
                findings.append(QCFinding(
                    type="gender_mismatch",
                    severity="severe",
                    description=f"女性患者使用男性专用药品：{drug_name}",
                    suggestion="请确认用药是否与患者性别匹配",
                    entities=[drug_name]
                ))

        # 检查诊断（疾病名）
        for disease in entities.diseases:
            disease_name = disease.std_name
            if gender == "male" and any(kw in disease_name for kw in FEMALE_ONLY_TESTS):
                findings.append(QCFinding(
                    type="gender_mismatch",
                    severity="severe",
                    description=f"男性患者涉及女性特有诊断：{disease_name}",
                    suggestion="请确认诊断是否与患者性别匹配",
                    entities=[disease_name]
                ))
            elif gender == "female" and any(kw in disease_name for kw in MALE_ONLY_TESTS):
                findings.append(QCFinding(
                    type="gender_mismatch",
                    severity="severe",
                    description=f"女性患者涉及男性特有诊断：{disease_name}",
                    suggestion="请确认诊断是否与患者性别匹配",
                    entities=[disease_name]
                ))

        return findings

    @staticmethod
    def check_dosage_abnormal(text: str) -> list[QCFinding]:
        """剂量异常检测"""
        findings = []

        # 常见剂量模式
        dosage_patterns = [
            (r'(\d+\.?\d*)\s*m?g\s*(?:tid|bid|qd|一天[一二三]次|每日[一二三]次)', "剂量"),
            (r'(\d+\.?\d*)\s*ml', "体积"),
        ]

        # 可疑的高剂量
        suspicious_doses = [
            (r'(\d{4,})\s*m?g', 1000, "剂量偏高（>1000mg），请确认"),
            (r'(\d{3,})\s*ml', 100, "单次体积偏大，请确认"),
        ]

        for pattern, _ in dosage_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                val = float(match)
                if val > 1000:
                    findings.append(QCFinding(
                        type="dosage_abnormal",
                        severity="warning",
                        description=f"剂量偏高：{match}mg",
                        suggestion="请确认剂量是否合理",
                    ))

        return findings

    @staticmethod
    def check_symptom_disease_mismatch(
        diseases: list, symptoms: list, kg_queries=None
    ) -> list[QCFinding]:
        """
        症状-诊断匹配检测（通过 KG 验证）

        关键点：多诊断共存时，病历症状可能属于其中某个疾病，
        不应因「症状没匹配到本疾病」就对每个疾病都报错。
        只有当症状整体未被任何确诊疾病的典型症状覆盖时，才提示记录不完整。
        """
        findings = []
        if not kg_queries or not diseases or not symptoms:
            return findings

        # 预取：每个确诊疾病的 KG 典型症状
        disease_kg = {}
        for disease in diseases:
            disease_kg[disease.std_name] = kg_queries.get_symptoms(disease.std_name) or []

        patient_syms = [s.std_name for s in symptoms]

        for disease in diseases:
            dname = disease.std_name
            kg_syms = disease_kg.get(dname, [])
            if not kg_syms:
                continue  # KG 未收录该疾病 → 跳过，不误报
            if not patient_syms:
                continue

            # 病历症状是否能匹配到本疾病的典型症状
            matched = [ps for ps in patient_syms
                       if any(ps in ks or ks in ps for ks in kg_syms)]
            if matched:
                continue  # 有匹配 → 正常

            # 病历症状是否整体上属于「其它确诊疾病」（多诊断共存场景）
            all_other = set()
            for other, ks in disease_kg.items():
                if other != dname:
                    for k in ks:
                        all_other.add(k)
            matched_else = [ps for ps in patient_syms
                            if any(ps in k or k in ps for k in all_other)]

            if matched_else:
                description = (
                    f"已确诊【{dname}】，但病历症状（{'、'.join(patient_syms)}）"
                    f"更接近其他诊断的典型表现，可能为多诊断共存或记录不全，请核实"
                )
                suggestion = "如属多诊断共存，建议分别记录各诊断对应的症状；否则补充该疾病的典型症状"
            else:
                description = (
                    f"已确诊【{dname}】，但病历症状（{'、'.join(patient_syms)}）"
                    f"未被知识图谱中【{dname}】的典型症状覆盖，请核实病历记录完整性"
                )
                suggestion = "建议补充【" + dname + "】的典型症状描述，或复核诊断"

            findings.append(QCFinding(
                type="symptom_disease_mismatch",
                severity="info",
                description=description,
                suggestion=suggestion,
                entities=[dname]
            ))

        return findings

    @staticmethod
    def check_missing_info(text: str, entities: StructuredRecord) -> list[QCFinding]:
        """信息完整性检查"""
        findings = []
        text_lower = text

        # 年龄缺失
        if not re.search(r'\d+\s*岁', text_lower):
            findings.append(QCFinding(
                type="missing_info",
                severity="warning",
                description="未记录患者年龄",
                suggestion="建议补充患者年龄信息，辅助用药剂量判断"
            ))

        # 诊断缺失
        if not entities.diseases:
            findings.append(QCFinding(
                type="missing_info",
                severity="warning",
                description="未识别到诊断信息",
                suggestion="请确认是否已记录诊断结果"
            ))

        # 用药无剂量
        if entities.drugs and not re.search(r'\d+\.?\d*\s*m?g', text_lower):
            findings.append(QCFinding(
                type="missing_info",
                severity="warning",
                description="药品缺少剂量信息",
                suggestion="建议补充药品剂量，如「二甲双胍0.5g tid」"
            ))

        return findings

    def run_all(self, text: str, entities: StructuredRecord, kg_queries=None) -> list[QCFinding]:
        """运行所有规则"""
        findings = []
        findings.extend(self.check_gender_mismatch(text, entities))
        findings.extend(self.check_dosage_abnormal(text))
        findings.extend(self.check_symptom_disease_mismatch(entities.diseases, entities.symptoms, kg_queries))
        findings.extend(self.check_missing_info(text, entities))

        # 去重：相同 type + description 的只保留一条
        seen = set()
        deduped = []
        for f in findings:
            key = (f.type, f.description)
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped