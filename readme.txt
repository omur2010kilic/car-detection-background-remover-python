TUR:
==================================================================
                ARKA PLAN SİLİCİ API (YOLOv8 + FastAPI)
==================================================================

Bu uygulama, bir resimdeki en büyük nesneyi tespit eder ve geri kalan
arka planı şeffaf hale getirir. YOLOv8 Segmentasyon modeli kullanır.
Çıktı olarak şeffaf arka planlı PNG resmi döner.

------------------------------------------------------------
🧠 ÖZELLİKLER
------------------------------------------------------------
- En büyük nesneyi otomatik seçer
- Geri kalan tüm alanları siler (şeffaf PNG olarak)
- Maske kenarlarını yumuşatır (Gaussian Blur + Morphology)
- YOLOv8 segmentasyon kullanır
- FastAPI ile web servisi olarak sunulur
- Swagger UI üzerinden kolay test imkanı

------------------------------------------------------------
🔧 GEREKSİNİMLER
------------------------------------------------------------
Python 3.8+ kurulu olmalıdır.

Gerekli kütüphaneleri yüklemek için:

    pip install fastapi uvicorn pillow numpy opencv-python ultralytics

------------------------------------------------------------
🚀 UYGULAMAYI ÇALIŞTIRMA
------------------------------------------------------------

1. Bu klasörde `main.py` dosyasının olduğundan emin ol.
2. Aşağıdaki komutla API'yi başlat:

    uvicorn main:app --reload

3. Tarayıcıdan Swagger arayüzüne git:

    http://127.0.0.1:8000/docs

4. `/remove_background` endpoint'ine gir, bir resim (.jpg/.png) yükle.
5. Çıktı olarak arka planı şeffaf PNG döner.

------------------------------------------------------------
🗂️ PROJE YAPISI
------------------------------------------------------------

- main.py        → Uygulamanın FastAPI kodları
- README.txt     → Bu dosya
- yolov8n-seg.pt → Segmentasyon modeli (varsayılan YOLOv8)

Kendi modelini kullanacaksan `main.py` içinde şu satırı değiştir:

    model = YOLO("yolov8n-seg.pt")
            ↓
    model = YOLO("runs/segment/train/weights/best.pt")

------------------------------------------------------------
🎯 İPUÇLARI
------------------------------------------------------------

- Bu uygulama sadece yüklenen resimler için çalışır.
- Gerçek zamanlı kamera görüntüsü için ayrı bir uygulama gerekir.
- Çıktı dosyası şeffaf PNG'dir, sosyal medyada kullanılabilir.

------------------------------------------------------------
📞 Kendinde Geliştirebilirsin!
------------------------------------------------------------

- Daha gelişmiş kullanım için `class name filter`, `webcam` desteği, 
  ya da `arka planı özel bir resimle değiştirme` eklenebilir.
- İsteğe bağlı olarak OpenCV + Tkinter arayüzü entegre edilebilir.

------------------------------------------------------------
🔥 HAZIRLAYAN
------------------------------------------------------------

Arka Plan Silici API — Ömür Efes



ENG: 
==================================================================
                   BACKGROUND REMOVAL API (YOLOv8 + FastAPI)
==================================================================

This application detects the largest object in an image and removes
the background by making it transparent. It uses YOLOv8 segmentation
model and outputs a PNG image with transparent background.

------------------------------------------------------------
🧠 FEATURES
------------------------------------------------------------
- Automatically selects the largest object
- Removes all other areas (returns transparent PNG)
- Smooths mask edges (Gaussian Blur + Morphology)
- Uses YOLOv8 segmentation
- Serves as a web API via FastAPI
- Easy testing via Swagger UI

------------------------------------------------------------
🔧 REQUIREMENTS
------------------------------------------------------------
Python 3.8+ is required.

Install dependencies with:

    pip install fastapi uvicorn pillow numpy opencv-python ultralytics

------------------------------------------------------------
🚀 RUNNING THE APPLICATION
------------------------------------------------------------

1. Make sure `main.py` is in the current folder.
2. Start the API with:

    uvicorn main:app --reload

3. Open your browser and go to Swagger UI:

    http://127.0.0.1:8000/docs

4. Use the `/remove_background` endpoint to upload an image (.jpg/.png).
5. The output will be a transparent background PNG.

------------------------------------------------------------
🗂️ PROJECT STRUCTURE
------------------------------------------------------------

- main.py        → FastAPI application code
- README.txt     → This file
- yolov8n-seg.pt → Segmentation model (default YOLOv8)

To use your own model, change this line in `main.py`:

    model = YOLO("yolov8n-seg.pt")
            ↓
    model = YOLO("runs/segment/train/weights/best.pt")

------------------------------------------------------------
🎯 TIPS
------------------------------------------------------------

- This app works only for uploaded images.
- For real-time camera input, a separate application is needed.
- Output file is transparent PNG, ready for social media use.

------------------------------------------------------------
📞 IMPROVE YOURSELF!
------------------------------------------------------------

- You can add `class name filtering`, `webcam support`, or 
  `custom background replacement`.
- Optionally integrate OpenCV + Tkinter for GUI.

------------------------------------------------------------
🔥 CREATED BY
------------------------------------------------------------

Background Removal API — Ömür Efes
