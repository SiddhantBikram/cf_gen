from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
from configs import *
import pickle

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

dir = os.path.join(inpaint_dir, 'train', '02_wheeled vehicle')

caps = []

for img in os.listdir(dir):
    image = Image.open(os.path.join(dir, img))
    text = ""
    inputs = processor(images=image, text=text, return_tensors="pt")
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    caps.append(generated_caption)

with open(os.path.join(root_dir, 'caps_car'), 'wb') as fp:
    pickle.dump(caps, fp)
