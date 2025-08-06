from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2

app = FastAPI()

# YOLOv8 Segmentasyon Modeli (yolov8n-seg.pt gibi bir model)
model = YOLO("yolov8n-seg.pt")  # veya kendi segmentasyon modelin

@app.post("/remove_background")
async def remove_background(file: UploadFile = File(...)):
    # Resim kontrolü
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir resim dosyası yükleyin.")

    try:
        # Görseli oku
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # YOLO segmentasyonunu uygula
        results = model(img_np)
        result = results[0]

        # Eğer maske yoksa hata ver
        if result.masks is None or result.masks.data.shape[0] == 0:
            raise HTTPException(status_code=404, detail="Resimde nesne bulunamadı.")

        # En büyük maskeyi seç
        masks = result.masks.data.cpu().numpy()
        largest_area = 0
        largest_mask = None

        for mask in masks:
            area = np.sum(mask)
            if area > largest_area:
                largest_area = area
                largest_mask = mask

        if largest_mask is None:
            raise HTTPException(status_code=404, detail="Geçerli bir nesne bulunamadı.")

        # Orijinal resim boyutuna göre maskeyi yeniden boyutlandır
        h, w = img_np.shape[:2]
        mask_resized = cv2.resize(largest_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Maskeyi 0-255 arası değerlere çevir
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        # 🎯 KÖŞELERİ DÜZELT: Gaussian blur + morphology
        blurred = cv2.GaussianBlur(mask_uint8, (21, 21), 0)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        # RGBA formatına geç, alfa kanalını uygula
        img_rgba = img.convert("RGBA")
        img_np_rgba = np.array(img_rgba)
        img_np_rgba[..., 3] = cleaned

        # PNG olarak kaydet
        output_img = Image.fromarray(img_np_rgba)
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Arka plan silici kral gibi çalışıyor 😎"}
