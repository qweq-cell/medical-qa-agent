"""
质控报告生成（Markdown 格式）
"""
from src.models.schemas import QCReport, QCFinding


class QCReporter:
    """质控报告生成器"""

    @staticmethod
    def generate_markdown(report: QCReport) -> str:
        """生成 Markdown 格式报告"""
        lines = []

        # 标题
        lines.append("# 🏥 病历质控报告")
        lines.append("")
        lines.append(f"> 生成时间：自动分析")
        lines.append("")

        # 摘要
        s = report.summary
        if s.passed:
            status = "✅ **通过** — 未发现严重问题"
        else:
            severe = s.severe_count
            warning = s.warning_count
            status = f"⚠️ **发现 {severe} 个严重问题、{warning} 个警告**"
        lines.append(f"## 质控摘要")
        lines.append(f"**状态：** {status}")
        lines.append(f"**发现总数：** {s.total_findings}")
        lines.append(f"| 严重程度 | 数量 |")
        lines.append(f"|:---------|:----:|")
        lines.append(f"| 🔴 严重 | {s.severe_count} |")
        lines.append(f"| 🟡 警告 | {s.warning_count} |")
        lines.append(f"| 🔵 提示 | {s.info_count} |")
        lines.append("")

        # 结构化实体
        lines.append("## 📋 结构化实体")
        e = report.entities
        if e.diseases:
            lines.append("### 疾病诊断")
            for d in e.diseases:
                lines.append(f"- {d.name}（标准化：{d.std_name}）")
            lines.append("")
        if e.symptoms:
            lines.append("### 症状")
            for s in e.symptoms:
                lines.append(f"- {s.name}（标准化：{s.std_name}）")
            lines.append("")
        if e.drugs:
            lines.append("### 药品")
            for d in e.drugs:
                lines.append(f"- {d.name}（标准化：{d.std_name}）")
            lines.append("")
        if e.tests:
            lines.append("### 检查项目")
            for t in e.tests:
                lines.append(f"- {t.name}（标准化：{t.std_name}）")
            lines.append("")
        if e.departments:
            lines.append("### 科室")
            for d in e.departments:
                lines.append(f"- {d.name}")
            lines.append("")

        # 质控发现
        if report.findings:
            lines.append("## 🔍 质控发现")
            lines.append("")
            for i, finding in enumerate(report.findings, 1):
                severity_icon = {"severe": "🔴", "warning": "🟡", "info": "🔵"}
                icon = severity_icon.get(finding.severity, "⚪")
                lines.append(f"### {i}. {icon} {finding.description}")
                lines.append(f"- **类型：** {finding.type}")
                lines.append(f"- **严重程度：** {finding.severity}")
                if finding.suggestion:
                    lines.append(f"- **建议：** {finding.suggestion}")
                if finding.entities:
                    lines.append(f"- **涉及实体：** {'、'.join(finding.entities)}")
                lines.append("")
        else:
            lines.append("## ✅ 质控结果")
            lines.append("未发现质控问题，病历书写规范。")
            lines.append("")

        # 原文
        lines.append("---")
        lines.append(f"**原始病历：**")
        lines.append(f"> {report.original_text}")

        return "\n".join(lines)