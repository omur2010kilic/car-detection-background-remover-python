from rembg import remove
from PIL import Image

input_path = 'girdi.jpg'
cikisfoto_path = 'cikti.png'  

with open(input_path, 'rb') as i:
    input_image = i.read()

    output_image = remove(input_image)

    with open(cikisfoto_path, 'wb') as o:
        o.write(output_image)

print("Background was deleted.", cikisfoto_path)
input("Çikmak için Enter'a bas...")