"""
OCR 引擎（PaddleOCR）：病历图片 → 文本

- 懒加载：首次调用才初始化 PaddleOCR（会下载模型，约几十 MB）
- 降级：未安装 paddleocr 时 available=False，调 recognize 返回 None
"""
import os
import sys


class OCREngine:
    def __init__(self, use_angle_cls: bool = True, lang: str = "ch"):
        self._use_angle_cls = use_angle_cls
        self._lang = lang
        self._ocr = None
        self._available = None

    @property
    def available(self) -> bool:
        """PaddleOCR 是否可用（依赖已安装且能导入）"""
        if self._available is None:
            try:
                import paddleocr  # noqa
                self._available = True
            except Exception as e:
                import traceback
                print(f"[OCR] PaddleOCR 导入失败，原因: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print("[OCR] 请确认已安装: pip install paddlepaddle==2.6.2 paddleocr==2.7.3", file=sys.stderr)
                self._available = False
        return self._available

    def _get_engine(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            # 首次调用会下载检测/识别模型，之后缓存
            self._ocr = PaddleOCR(use_angle_cls=self._use_angle_cls, lang=self._lang, show_log=False)
        return self._ocr

    def recognize(self, image_path: str) -> str | None:
        """识别图片文本；未安装 PaddleOCR 或识别失败返回 None"""
        if not self.available:
            return None
        try:
            engine = self._get_engine()
            result = engine.ocr(image_path, cls=True)
            lines = []
            # 兼容 PaddleOCR 2.x / 3.x 返回结构
            for page in (result or []):
                if not page:
                    continue
                for item in page:
                    try:
                        txt = item[1][0]
                        if txt:
                            lines.append(txt)
                    except (IndexError, TypeError):
                        continue
            return "\n".join(lines).strip() or None
        except Exception as e:
            print(f"[OCR] 识别失败: {e}", file=sys.stderr)
            return None
