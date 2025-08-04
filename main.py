from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from rembg import remove
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import logging

app = FastAPI()
model = YOLO('yolov8n.pt')  # Hızlı model (YOLOv8 nano)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>Araba Arka Plan Silici + YOLOv8</title></head>
        <body>
            <h2>Araba fotoğrafınızı yükleyin</h2>
            <form action="/remove-background/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <input type="submit" value="Gönder">
            </form>
        </body>
    </html>
    """

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    try:
        
        input_bytes = await file.read()
        img = Image.open(BytesIO(input_bytes)).convert("RGB")

        i
        results = model(img)
        boxes = [
            box.xyxy[0].tolist()
            for r in results
            for box in r.boxes
            if int(box.cls[0]) == 2  # 2 = car (COCO class ID)
        ]

        if not boxes:
            return JSONResponse(status_code=404, content={"error": "Fotoğrafta araba bulunamadı."})

    
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img.crop((x1, y1, x2, y2))

    
        with BytesIO() as buffer:
            cropped.save(buffer, format='PNG')
            output = remove(buffer.getvalue())

        return StreamingResponse(BytesIO(output), media_type="image/png")

    except Exception as e:
        logging.exception("Hata oluştu:")
        return JSONResponse(status_code=500, content={"error": f"Bir hata oluştu: {str(e)}"})
