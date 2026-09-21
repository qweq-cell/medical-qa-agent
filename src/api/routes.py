"""
FastAPI 路由定义
"""
import os
import tempfile
import time
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from src.agent.pipeline import MedicalPipeline
from src.models.schemas import QCReport
from src.ocr.ocr_engine import OCREngine

router = APIRouter(prefix="/api", tags=["medguardian"])

# 全局 Agent 实例（由 main.py 注入；P2 起为 MedicalPipeline，按 PIPELINE_MODE 决定管线）
agent: MedicalPipeline = None
_ocr = OCREngine()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="病历文本")


class AnalyzeResponse(BaseModel):
    success: bool
    report: QCReport
    latency_ms: int


class AnalyzeImageResponse(BaseModel):
    success: bool
    report: QCReport
    ocr_text: str = ""
    latency_ms: int


@router.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "MedGuardian"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """分析病历文本"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start = time.time()
    report = agent.analyze(request.text)
    latency = int((time.time() - start) * 1000)

    return AnalyzeResponse(success=True, report=report, latency_ms=latency)


@router.post("/analyze_image", response_model=AnalyzeImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    """分析病历图片：OCR 识别 → 质控流程"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # 保存上传图片到临时文件
    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if not _ocr.available:
            raise HTTPException(
                status_code=501,
                detail="PaddleOCR 未安装，请运行 `pip install paddlepaddle paddleocr` 后重启服务",
            )

        text = _ocr.recognize(tmp_path)
        if not text:
            raise HTTPException(status_code=422, detail="未能从图片中识别出文字，请换一张更清晰的图片")

        start = time.time()
        report = agent.analyze(text)
        latency = int((time.time() - start) * 1000)
        return AnalyzeImageResponse(success=True, report=report, ocr_text=text, latency_ms=latency)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
