import os

from PIL import Image

all_dir = os.listdir("my_photos")

for dir in all_dir:
    im = Image.open("my_photos/" + dir)
    size = (224, 224)
    im = im.resize(size)
    im = im.save("custom/"+dir)