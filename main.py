from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI(title="YOLO + Rembg API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO ve Rembg modelleri
yolo_model = YOLO("yolo11n-seg.pt")
rembg_session = new_session(model_name="u2netp")

# Görsel okuma
async def read_image(file: UploadFile) -> Image.Image:
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Lütfen bir görsel dosyası yükleyin.")
    contents = await file.read()
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.")

# En büyük aracı tespit et
def detect_largest_vehicle(image: Image.Image) -> Image.Image:
    image_np = np.array(image)
    results = yolo_model(image_np)
    if not results or len(results[0].boxes) == 0:
        raise HTTPException(status_code=404, detail="Araç bulunamadı.")
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    largest_idx = areas.index(max(areas))
    x1, y1, x2, y2 = map(int, boxes[largest_idx])
    return image.crop((x1, y1, x2, y2))

# Arka plan kaldır
def remove_background(image: Image.Image) -> io.BytesIO:
    output = remove(image, session=rembg_session)
    img_bytes = io.BytesIO()
    output.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

# API endpoint
@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    image = await read_image(file)
    cropped = detect_largest_vehicle(image)
    result = remove_background(cropped)
    return StreamingResponse(
        result,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="processed.png"'}
    )

# Static dosyaları kök URL olarak sun
app.mount("/", StaticFiles(directory="static", html=True), name="static")
