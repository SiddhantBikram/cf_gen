from PIL import Image, ImageDraw, ImageFilter
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import easyocr
from lama_loader import *
from torchvision import transforms

path = 'C:/Users/siddh/Desktop/Humour/jk83t88ue7q61.jpg'
image_dim = 512
img_pil = Image.open(path)

reader = easyocr.Reader(['en']) 
result = reader.readtext(path, paragraph=False)	

mask = Image.new(mode="RGB", size=(np.array(img_pil).shape[1], np.array(img_pil).shape[0]))
draw = ImageDraw.Draw(mask)
text = ''

for i in result:
    text = text + i[1] + ' '
    point_0 = (i[0][0][0], i[0][0][1])
    point_1 = (i[0][2][0], i[0][2][1])
    draw.rectangle([point_0, point_1], fill = 'white')

print(text)

mask = mask.convert("L")

trans = transforms.Compose(
    [
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
    ]
)

img_pil = trans(img_pil)
mask = mask.resize((image_dim,image_dim), Image.Resampling.LANCZOS)
mask.show()

inpaint_model = init_inpaint()
bg = inpaint(img_pil, mask, inpaint_model)[:,:,::-1]

bg = Image.fromarray(bg, mode = "RGB")

#Edge removal
img_cv = np.array(bg)
img_cv = cv.resize(img_cv, (image_dim, image_dim), interpolation = cv2.INTER_AREA)
edges = cv.Canny(img_cv,100,200)
y_nonzero, x_nonzero = np.nonzero(edges)
final = bg.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))

Image.fromarray(edges).show()

