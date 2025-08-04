from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from rembg import remove
from io import BytesIO

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
        <head>
            <title>Arka Plan Silici</title>
        </head>
        <body>
            <h2>Araba fotoğrafınızı yükleyin, arka plan silinsin</h2>
            <form action="/remove-background/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Gönder">
            </form>
        </body>
    </html>
    """
    return html_content


@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    input_bytes = await file.read()
    output_bytes = remove(input_bytes)
    return StreamingResponse(BytesIO(output_bytes), media_type="image/png")
#elime saglik