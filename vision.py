import logging
import os
import time

import cv2

from utils import bundle_dir, configure_runtime_env

logger = logging.getLogger(__name__)

_OCR_ALLOWLIST = "0123456789.,KMBkmb"

_SUFFIX_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def configure_easyocr_paths():
    if os.environ.get("EASYOCR_MODULE_PATH"):
        return

    data_dir = os.path.join(bundle_dir(), "easyocr_data")
    if not os.path.isdir(os.path.join(data_dir, "model")):
        return

    os.environ["EASYOCR_MODULE_PATH"] = data_dir
    logger.debug(f"EasyOCR weights directory set to {data_dir}")


def parse_loot_number(text: str) -> str:
    if not text:
        return "0"

    cleaned = (
        text.strip().lower().replace(",", "").replace(" ", "").replace("+", "")
    )
    if not cleaned:
        return "0"

    suffix = cleaned[-1]
    if suffix in _SUFFIX_MULTIPLIERS:
        core = cleaned[:-1]
        try:
            value = float(core) * _SUFFIX_MULTIPLIERS[suffix]
        except ValueError:
            digits = "".join(ch for ch in core if ch.isdigit())
            value = float(digits or "0") * _SUFFIX_MULTIPLIERS[suffix]
        return str(int(round(value)))

    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return digits or "0"


class VisionManager:

    def __init__(self, config):
        self.config = config
        self.model = None
        self.reader = None
        self.last_inference_ms = 0.0
        self.last_ocr_ms = 0.0
        self.use_gpu = False
        self.device = "cpu"
        self.device_name = "CPU"
    
    def load(self):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("PyTorch is required for YOLO and EasyOCR") from e

        cuda_available = torch.cuda.is_available()
        if self.config.gpu is True and not cuda_available:
            raise RuntimeError(
                "GPU acceleration was requested, but this PyTorch installation "
                "cannot access CUDA. Install the CUDA build from requirements-gpu.txt."
            )
        self.use_gpu = cuda_available if self.config.gpu is None else self.config.gpu
        if self.use_gpu:
            self.device = "cuda:0"
            self.device_name = torch.cuda.get_device_name(0)
            logger.info("Vision device: %s (CUDA, FP16 inference)", self.device_name)
        else:
            self.device = "cpu"
            self.device_name = "CPU"
            logger.info(
                "Vision device: CPU%s",
                " (CUDA unavailable in this PyTorch build)" if not cuda_available else "",
            )

        if self.model is None:
            try:
                configure_runtime_env()
                from ultralytics import YOLO

                self.model = YOLO(self.config.model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO model from '{self.config.model_path}': {e}")

        if self.reader is None:
            try:
                configure_easyocr_paths()
                import easyocr

                self.reader = easyocr.Reader(["en"], gpu=self.use_gpu)
            except Exception as e:
                raise RuntimeError(f"Failed to load EasyOCR reader: {e}")
    
    def detect(self, frame, class_names, conf=None):
        details = self.detect_with_details(frame, class_names, conf)
        return {
            cls: [item["center"] for item in items]
            for cls, items in details.items()
        }

    def detect_with_details(self, frame, class_names, conf=None):
        if self.model is None:
            self.load()

        if conf is None:
            conf = self.config.conf

        if self.use_gpu:
            import torch

            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        result = self.model.predict(
            source=frame,
            conf=conf,
            device=self.device,
            quantize=16 if self.use_gpu else None,
            verbose=False,
        )
        if self.use_gpu:
            torch.cuda.synchronize()
        self.last_inference_ms = (time.perf_counter() - inference_started) * 1000.0
        detections = {}

        for box in result[0].boxes:
            class_id = int(box.cls.item())
            class_name = str(self.model.names[class_id])

            if class_name in class_names:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = x1 + (x2 - x1) / 2
                cy = y1 + (y2 - y1) / 2

                if class_name not in detections:
                    detections[class_name] = []
                detections[class_name].append({
                    "center": (cx, cy),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf.item())
                })

        return detections

    def warmup(self, frame):
        self.detect_with_details(frame, ())

        started = time.perf_counter()
        try:
            self.reader.readtext(
                self._preprocess(frame), detail=0, allowlist=_OCR_ALLOWLIST
            )
        finally:
            self.last_ocr_ms = (time.perf_counter() - started) * 1000.0

    def _preprocess(self, region):
        h, w = region.shape[:2]
        if h and h < 64:
            scale = 64.0 / h
            region = cv2.resize(
                region,
                (max(1, int(w * scale)), 64),
                interpolation=cv2.INTER_CUBIC,
            )
        return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    def read_number(self, frame, bbox):
        started = time.perf_counter()
        try:
            return self._read_number(frame, bbox)
        finally:
            self.last_ocr_ms = (time.perf_counter() - started) * 1000.0

    def _read_number(self, frame, bbox):
        if self.reader is None:
            self.load()

        x1, y1, x2, y2 = bbox
        region = frame[y1:y2, x1:x2]

        if region.size == 0:
            return "0"

        ocr_result = self.reader.readtext(
            self._preprocess(region), detail=1, allowlist=_OCR_ALLOWLIST
        )

        parts = []
        dropped = []
        for item in ocr_result or []:
            text = str(item[1]).strip()
            if not text:
                continue
            digits = sum(ch.isdigit() for ch in text)
            if digits and digits / len(text.replace(" ", "")) >= 0.6:
                parts.append(text)
            else:
                dropped.append(text)

        raw = " ".join(parts)
        value = parse_loot_number(raw)
        logger.debug(
            f"OCR bbox={(x1, y1, x2, y2)} raw={raw!r} dropped={dropped!r} "
            f"-> {value}"
        )
        return value
