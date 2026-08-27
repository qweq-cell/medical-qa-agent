"""
MedGuardian Streamlit 前端
"""
import streamlit as st
import requests
import json

st.set_page_config(
    page_title="MedGuardian — 病历质控 Agent",
    page_icon="🏥",
    layout="wide",
)

API_URL = "http://localhost:8000"

st.title("🏥 MedGuardian — 智能病历质控 Agent")
st.markdown("> 输入病历文本（或上传图片）→ AI 自动提取实体 → 检测临床逻辑矛盾 → 输出质控报告")


def show_report(report: dict, latency: int):
    """展示质控报告（文本分析与图片分析共用）"""
    # 摘要
    summary = report["summary"]
    if summary["passed"]:
        st.success(f"✅ 质控通过 (耗时 {latency}ms)")
    else:
        st.error(f"⚠️ 发现 {summary['severe_count']} 个严重问题 (耗时 {latency}ms)")

    # LLM 质控报告（Agent 模式）
    if report.get("llm_report"):
        st.subheader("🤖 LLM 质控报告")
        st.markdown(report["llm_report"])

    # 统计卡片
    c1, c2, c3, c4 = st.columns(4)
    ent = report["entities"]
    c1.metric("疾病", len(ent.get("diseases", [])))
    c2.metric("症状", len(ent.get("symptoms", [])))
    c3.metric("药品", len(ent.get("drugs", [])))
    c4.metric("检查", len(ent.get("tests", [])))

    # 质控发现
    st.subheader("🔍 质控发现")
    if report["findings"]:
        for finding in report["findings"]:
            sev = finding["severity"]
            icon = {"severe": "🔴", "warning": "🟡", "info": "🔵"}
            st.markdown(f"{icon.get(sev, '⚪')} **{finding['description']}**")
            if finding.get("suggestion"):
                st.caption(f"💡 建议：{finding['suggestion']}")
            if finding.get("entities"):
                st.caption(f"涉及：{'、'.join(finding['entities'])}")
            st.divider()
    else:
        st.info("未发现质控问题 ✅")

    # 结构化实体
    with st.expander("📋 查看结构化实体"):
        for cat, label in [("diseases", "疾病"), ("symptoms", "症状"),
                           ("drugs", "药品"), ("tests", "检查项目")]:
            items = ent.get(cat, [])
            if items:
                st.markdown(f"**{label}：**")
                for item in items:
                    st.markdown(f"- {item['name']} → `{item['std_name']}`")

    # 原始文本
    with st.expander("📄 查看原始文本"):
        st.text(report["original_text"])


# 侧边栏
with st.sidebar:
    st.header("📋 使用说明")
    st.markdown("""
    1. 在文本框**输入病历文本**，或**上传病历图片**（OCR 识别）
    2. 点击「分析」
    3. 查看结构化实体、质控发现和 LLM 报告

    **支持检测：**
    - 性别矛盾（男→妇科检查）
    - 剂量异常
    - 诊断-症状匹配
    - 信息完整性
    """)

    st.header("🔗 系统状态")
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=3)
        if resp.status_code == 200:
            st.success("✅ 后端服务运行中")
        else:
            st.error("❌ 后端异常")
    except Exception:
        st.warning("⚠️ 后端未连接（请先启动 FastAPI）")

    st.header("📝 示例文本")
    sample_text = """患者张伟，男，67岁，因口渴多饮、多尿、体重下降3个月就诊。既往2型糖尿病病史5年，高血压病史3年。查体：空腹血糖8.5mmol/L，糖化血红蛋白7.2%。诊断：2型糖尿病，高血压。用药：二甲双胍0.5g tid，氨氯地平5mg qd。"""
    if st.button("加载示例病历"):
        st.session_state["input_text"] = sample_text

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 输入病历文本")
    text = st.text_area(
        "在此粘贴病历文本：",
        key="input_text",
        height=250,
        placeholder="例如：患者张伟，男，67岁，因口渴多饮就诊。诊断：2型糖尿病。用药：二甲双胍0.5g tid。",
    )
    analyze_btn = st.button("🔍 分析", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📷 或上传病历图片")
    img_file = st.file_uploader("上传病历图片（支持 JPG/PNG）", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        # 预览图片
        try:
            st.image(img_file, caption="上传的图片", use_column_width=True)
        except Exception:
            pass
        analyze_img_btn = st.button("🔍 识别图片并质控", use_container_width=True)
    else:
        analyze_img_btn = False

with col2:
    st.header("📊 分析结果")

    # 文本分析
    if analyze_btn and text.strip():
        with st.spinner("正在分析..."):
            try:
                resp = requests.post(f"{API_URL}/api/analyze",
                                     json={"text": text.strip()}, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    show_report(data["report"], data["latency_ms"])
                else:
                    st.error(f"分析失败: {resp.status_code}")
                    st.json(resp.json())
            except Exception as e:
                st.error(f"请求失败: {e}")
                st.info("请确保 FastAPI 后端已启动: `python -m src.main`")
    elif analyze_btn:
        st.warning("请输入病历文本")

    # 图片分析
    if analyze_img_btn and img_file is not None:
        with st.spinner("OCR 识别中（首次需下载模型，稍等）..."):
            try:
                files = {"file": (img_file.name, img_file.getvalue(), "image/png")}
                resp = requests.post(f"{API_URL}/api/analyze_image",
                                     files=files, timeout=180)
                if resp.status_code == 200:
                    data = resp.json()
                    ocr_text = data.get("ocr_text", "")
                    with st.expander("📄 识别出的病历文本", expanded=True):
                        st.text(ocr_text)
                    show_report(data["report"], data["latency_ms"])
                else:
                    detail = ""
                    try:
                        detail = resp.json().get("detail", "")
                    except Exception:
                        pass
                    st.error(f"识别失败 ({resp.status_code}): {detail}")
                    st.info("若提示未安装 PaddleOCR，请运行 `pip install paddlepaddle paddleocr` 后重启后端")
            except Exception as e:
                st.error(f"识别失败: {e}")

# 底部
st.markdown("---")
st.markdown("MedGuardian v0.1.0 | 秋招项目 | 2026")
