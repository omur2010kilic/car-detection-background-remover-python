# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
from typing import Set, Tuple

app = FastAPI(
    title="YOLO11 Largest Vehicle Background Remover",
    version="1.0.0",
    description="En büyük aracı tespit edip arka planı kaldıran FastAPI servisi."
)

model = YOLO("yolo11n-seg.pt")

VEHICLE_KEYWORDS = {"car", "truck", "bus", "van", "motorcycle", "motorbike", "vehicle", "auto"}


# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def read_image_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(pil)
    return arr, pil


def get_vehicle_class_ids(model, keywords: Set[str]):
    names = getattr(model, "names", {})
    return [int(k) for k, v in names.items() if any(kw in str(v).lower() for kw in keywords)]


def run_inference(img_np: np.ndarray, conf: float = 0.25, iou: float = 0.45):
    return model(img_np, conf=conf, iou=iou)[0]


def extract_largest_vehicle_mask_from_seg(result, vehicle_class_ids, conf_threshold: float = 0.2):
    if not hasattr(result, "masks") or result.masks is None or result.masks.data is None:
        return None

    masks_np = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) > 0 else []
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) > 0 else []

    largest_mask = None
    largest_area = 0

    for i in range(masks_np.shape[0]):
        if i >= len(cls_ids):
            continue
        if int(cls_ids[i]) in vehicle_class_ids and (len(confs) == 0 or float(confs[i]) >= conf_threshold):
            m = masks_np[i] > 0.5
            area = np.sum(m)
            if area > largest_area:
                largest_area = area
                largest_mask = m

    return largest_mask if largest_area > 0 else None


def fallback_largest_vehicle_box(result, vehicle_class_ids, img_shape, conf_threshold: float = 0.2, padding: int = 6):
    H, W = img_shape[:2]
    if not hasattr(result, "boxes") or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    largest_area = 0
    best_box = None

    for i, box in enumerate(boxes):
        if int(cls_ids[i]) in vehicle_class_ids and float(confs[i]) >= conf_threshold:
            x1, y1, x2, y2 = box.astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W - 1, x2 + padding)
    y2 = min(H - 1, y2 + padding)

    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def refine_mask(mask_bool: np.ndarray, min_area: int = 500) -> np.ndarray:
    mask = (mask_bool.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if np.sum(mask_closed) == 0:
        return mask_closed

    blurred = cv2.GaussianBlur(mask_closed, (15, 15), 0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def apply_alpha_to_image(pil_img: Image.Image, alpha_uint8: np.ndarray) -> Image.Image:
    img_rgba = pil_img.convert("RGBA")
    img_arr = np.array(img_rgba)
    if alpha_uint8.shape[0] != img_arr.shape[0] or alpha_uint8.shape[1] != img_arr.shape[1]:
        alpha_uint8 = cv2.resize(alpha_uint8, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
    img_arr[..., 3] = alpha_uint8
    return Image.fromarray(img_arr)


def pil_to_stream_bytes(pil_img: Image.Image, fmt: str = "PNG") -> io.BytesIO:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ---------------------------
# Endpointler
# ---------------------------
@app.post("/remove_background")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir resim dosyası yükleyin.")

    try:
        image_bytes = await file.read()
        img_np, pil_img = read_image_from_bytes(image_bytes)
        vehicle_ids = get_vehicle_class_ids(model, VEHICLE_KEYWORDS)
        if not vehicle_ids:
            raise HTTPException(status_code=500, detail="Modelde araç sınıfı bulunamadı.")

        result = run_inference(img_np, conf=0.25, iou=0.45)
        vehicle_mask = extract_largest_vehicle_mask_from_seg(result, vehicle_ids, conf_threshold=0.25)

        if vehicle_mask is None:
            vehicle_mask = fallback_largest_vehicle_box(result, vehicle_ids, img_np.shape, conf_threshold=0.25, padding=8)

        if vehicle_mask is None:
            raise HTTPException(status_code=404, detail="Araç tespit edilemedi.")

        alpha_uint8 = refine_mask(vehicle_mask, min_area=400)
        out_img = apply_alpha_to_image(pil_img, alpha_uint8)
        buf = pil_to_stream_bytes(out_img, fmt="PNG")

        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")


@app.get("/")
async def root():
    return {"message": "YOLO11 en büyük araç arka plan kaldırma servisi çalışıyor!"}
