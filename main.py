from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from rembg import remove, new_session
from PIL import Image
import io
import numpy as np

app = FastAPI(
    title="YOLO11 + Rembg Arka Plan Kaldırma API",
    description="Araç tespit edilir (YOLO11) ve arka plan kaldırılır (Rembg, Birefnet-HRSOD).",
    version="1.0.0"
)

# -----------------------
# 1. YOLO Modelini Yükle
# -----------------------
def load_yolo_model(model_path="yolo11n-seg.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"YOLO modeli yüklenemedi: {e}")

yolo_model = load_yolo_model()

# -----------------------
# 2. Rembg Session (Model: birefnet-hrsod)
# -----------------------
rembg_session = new_session(model_name="birefnet-hrsod")

# -----------------------
# 3. Görseli Oku
# -----------------------
async def read_image(file: UploadFile) -> Image.Image:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir görsel dosyası yükleyin.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.")

# -----------------------
# 4. YOLO ile Araç Tespiti (En büyük araç crop alınır)
# -----------------------
def detect_largest_vehicle(image: Image.Image):
    image_np = np.array(image)
    results = yolo_model(image_np)
    
    if not results or len(results[0].boxes) == 0:
        raise HTTPException(status_code=404, detail="Araç bulunamadı.")
    
    # En büyük bounding box'ı bul
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
    largest_idx = areas.index(max(areas))
    largest_box = boxes[largest_idx]

    x1, y1, x2, y2 = map(int, largest_box)
    return image.crop((x1, y1, x2, y2))  # Croplanmış araç resmi

# -----------------------
# 5. Rembg ile Arka Plan Kaldır
# -----------------------
def remove_background(image: Image.Image) -> bytes:
    try:
        output = remove(image, session=rembg_session)
        img_byte_arr = io.BytesIO()
        output.save(img_byte_arr, format="PNG")  # Şeffaf arka plan için PNG
        img_byte_arr.seek(0)
        return img_byte_arr
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arka plan kaldırılamadı: {e}")

# -----------------------
# 6. API Endpoint
# -----------------------
@app.post("/")
async def process_image(file: UploadFile = File(...)):
    image = await read_image(file)
    cropped_vehicle = detect_largest_vehicle(image)
    result = remove_background(cropped_vehicle)
    return StreamingResponse(result, media_type="image/png")
