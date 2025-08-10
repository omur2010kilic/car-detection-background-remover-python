from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

app = FastAPI()

# 📌 YOLO11 SEGMENTASYON modeli (maskeler için -seg şart)
model = YOLO("yolo11n-seg.pt")  # veya kendi model dosyan

@app.post("/remove_background")
async def remove_background(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir resim dosyası yükleyin.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # 🔍 YOLO tahmini
        results = model(img_np)
        result = results[0]

        mask_final = None

        # 🎯 Önce segmentasyon maskesi var mı bak
        if result.masks is not None and result.masks.data.shape[0] > 0:
            masks = result.masks.data.cpu().numpy()
            # En büyük maskeyi seç
            largest_mask = max(masks, key=lambda m: np.sum(m))
            mask_final = largest_mask.astype(np.uint8)

        # ❗ Maske yoksa bbox üzerinden maske üret
        if mask_final is None and len(result.boxes) > 0:
            h, w = img_np.shape[:2]
            mask_final = np.zeros((h, w), dtype=np.uint8)
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                mask_final[y1:y2, x1:x2] = 1  # dikdörtgen alanı doldur

        if mask_final is None:
            raise HTTPException(status_code=404, detail="Nesne tespit edilemedi.")

        # 📏 Maskeyi orijinal boyuta getir (gerekirse)
        h, w = img_np.shape[:2]
        mask_resized = cv2.resize(mask_final, (w, h), interpolation=cv2.INTER_NEAREST)

        # 🎨 0-255 arası uint8
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        # ✨ Kenar yumuşatma
        blurred = cv2.GaussianBlur(mask_uint8, (21, 21), 0)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        # 🖼 RGBA formatına geçir ve alfa uygula
        img_rgba = img.convert("RGBA")
        img_np_rgba = np.array(img_rgba)
        img_np_rgba[..., 3] = cleaned

        # 📦 PNG olarak döndür
        output_img = Image.fromarray(img_np_rgba)
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")


@app.get("/")
async def root():
    return {"message": "YOLO11 arka plan kaldırma servisi çalışıyor!"}
