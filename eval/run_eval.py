"""
MedGuardian 规则质控评测集 runner（Rule-only，无 Neo4j / 无 LLM）

用法:
    python eval/run_eval.py                # 运行并写 eval/report.md
    python eval/run_eval.py --json         # 仅打印 JSON 汇总

指标口径:
    recall      = 命中期望问题类型数 / 期望问题类型总数   （漏检越低越好）
    fp          = 实际抛出但非期望的问题类型数            （误报/噪声）
    precision   = detected / (detected + fp)
    clean-FP    = 期望为空(干净对照组)却抛出任一问题的病历数
    延迟        每次 analyze 毫秒：mean / p50 / p95
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ENTITY_DICT_FILE, EXTRA_ENTITIES          # noqa: E402
from src.ner.dict_matcher import DictMatcher                      # noqa: E402
from src.ner.entity_linking import EntityLinker                   # noqa: E402
from src.qc.rules import QCRules                                  # noqa: E402
from src.agent.tools import AgentTools                            # noqa: E402
from src.agent.orchestrator import MedGuardianAgent               # noqa: E402


def build_agent():
    matcher = DictMatcher(ENTITY_DICT_FILE, extra_entities=EXTRA_ENTITIES)
    linker = EntityLinker(ENTITY_DICT_FILE)
    tools = AgentTools(matcher=matcher, linker=linker, kg=None, rules=QCRules())
    return MedGuardianAgent(tools, llm=None)


def main():
    data = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    agent = build_agent()

    rows = []
    per_type = {}          # type -> {"exp":0, "hit":0}
    latencies = []
    for c in cases:
        text = c["text"]
        t0 = time.perf_counter()
        report = agent.analyze(text)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        exp = set(c.get("expected_types", []))
        raised = {f.type for f in report.findings}
        hit = exp & raised
        fp = raised - exp

        for t in exp:
            per_type.setdefault(t, {"exp": 0, "hit": 0})["exp"] += 1
            if t in raised:
                per_type[t]["hit"] += 1

        rows.append({
            "id": c["id"], "title": c["title"],
            "expected": sorted(exp), "raised": sorted(raised),
            "hit": sorted(hit), "fp": sorted(fp),
            "findings": len(report.findings),
            "severe": report.summary.severe_count,
            "warning": report.summary.warning_count,
            "info": report.summary.info_count,
            "latency_ms": round(ms, 1),
            "ok": (exp <= raised) and not fp,
        })

    total_exp = sum(len(set(c["expected_types"])) for c in cases)
    total_hit = sum(len(r["hit"]) for r in rows)
    total_fp = sum(len(r["fp"]) for r in rows)
    recall = total_hit / total_exp if total_exp else 1.0
    precision = total_hit / (total_hit + total_fp) if (total_hit + total_fp) else 1.0
    clean_fp = sum(1 for c, r in zip(cases, rows) if not c.get("expected_types") and r["raised"])
    all_ok = sum(1 for r in rows if r["ok"])
    lats = sorted(latencies)

    agg = {
        "cases": len(cases),
        "expected_instances": total_exp,
        "detected_instances": total_hit,
        "recall": round(recall, 3),
        "unexpected_findings": total_fp,
        "precision": round(precision, 3),
        "clean_cases": sum(1 for c in cases if not c.get("expected_types")),
        "clean_cases_with_fp": clean_fp,
        "fully_ok_cases": all_ok,
        "per_type_recall": {t: round(v["hit"] / v["exp"], 3) if v["exp"] else None
                            for t, v in sorted(per_type.items())},
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(statistics.median(latencies), 1),
            "p95": round(lats[int(len(lats) * 0.95) - 1], 1),
            "max": round(max(latencies), 1),
        },
    }

    if "--json" in sys.argv:
        print(json.dumps(agg, ensure_ascii=False, indent=2))
        return

    # markdown report
    lines = [
        "# MedGuardian 规则质控评测报告",
        "",
        f"- 数据集：`eval/cases.json`（{len(cases)} 例，合成埋点，见文件 meta）",
        f"- 模式：rule-only（无 Neo4j / 无 LLM）｜ 日期：{data['meta'].get('date','')}",
        "",
        "## 汇总",
        "",
        f"| 指标 | 值 |",
        "|---|---|",
        f"| 植入问题实例（期望） | {total_exp} |",
        f"| 检出 | {total_hit} |",
        f"| **Recall（检出率）** | **{agg['recall']:.1%}** |",
        f"| 误报（非期望抛出） | {total_fp} |",
        f"| **Precision** | **{agg['precision']:.1%}** |",
        f"| 干净对照组误报病历 | {clean_fp}/{agg['clean_cases']} |",
        f"| 完全正确病历（无漏检无误报） | {all_ok}/{len(cases)} |",
        f"| 延迟 mean/p50/p95(ms) | {agg['latency_ms']['mean']}/{agg['latency_ms']['p50']}/{agg['latency_ms']['p95']} |",
        "",
        "### 分类型检出率",
        "",
        "| 类型 | 期望 | 命中 | Recall |",
        "|---|---|---|---|",
    ]
    for t, v in sorted(per_type.items()):
        lines.append(f"| {t} | {v['exp']} | {v['hit']} | {v['hit']/v['exp']:.0%} |")
    lines += [
        "",
        "## 逐例明细",
        "",
        "| id | 期望 | 实际抛出 | 命中 | 误报 | 延迟ms | 完全正确 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} {r['title']} | {'、'.join(r['expected']) or '—'} | "
            f"{'、'.join(r['raised']) or '—'} | {'、'.join(r['hit']) or '—'} | "
            f"{'、'.join(r['fp']) or '—'} | {r['latency_ms']} | {'✅' if r['ok'] else '❌'} |"
        )
    lines.append("")
    (ROOT / "eval" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
