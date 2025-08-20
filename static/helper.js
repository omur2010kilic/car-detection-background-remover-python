document.addEventListener("DOMContentLoaded", () => {
    const sendButton = document.getElementById("sendButton");
    const fileInput = document.getElementById("fileInput");
    const titleInput = document.getElementById("titleInput");
    const statusText = document.getElementById("status");

    sendButton.addEventListener("click", async () => {
        statusText.textContent = "";
        if (!fileInput.files.length) {
            alert("Lütfen bir fotoğraf seçin!");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        if (titleInput.value) formData.append("title", titleInput.value);

        try {
            statusText.textContent = "Gönderiliyor...";
            const response = await fetch("http://127.0.0.1:8000/process", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error(`Sunucu hatası: ${response.status}`);

            const blob = await response.blob();
            const imgURL = URL.createObjectURL(blob);

            const newTab = window.open();
            newTab.document.write(`
                <html>
                    <head><title>Sonuç</title></head>
                    <body style="margin:0; display:flex; justify-content:center; align-items:center; height:100vh; background:#111;">
                        <img src="${imgURL}" style="max-width:100%; max-height:100%;">
                    </body>
                </html>
            `);

            statusText.textContent = "Başarılı!";
        } catch (error) {
            console.error(error);
            alert("Bir hata oluştu: " + error.message);
            statusText.textContent = "Hata oluştu.";
        }
    });
});
