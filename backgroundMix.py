import os
from PIL import Image
import random

file_path = "background_statistics.csv" 
keywords = [ "building", "Tree", "Parking space", "Road", "Grassland", "Exhibition hall"
    ]
synonyms_dict = {
    "building": ['Indoor environment', 'Residential building', 'Gas station', 'Structure/building', 'Port', 'Door', 'Architecture', 'Dealership/store', 'City', 'Glass door', 'Residential area', 'Countryside', 'House', 'Commercial district', 'Suburb', 'Glass window', 'Residence', 'Industrial area', 'Building/house', 'Window', 'Skyscraper', 'Ceiling', 'Roller shutter door', 'Commercial building', 'Modern architecture', 'Factory', 'Industrial building', 'Office building', 'High-rise building', 'Tower', 'Residential architecture'],
    "Tree": ['Tree', 'Shrub','Palm tree','Green tree','Tree'],
    "Parking space": ['Parking lot','Parking space','Parking line','Parking area','Race track','Warehouse','Underground parking lot','Garage'],
    "Road": ['Road', 'Street', 'Highway', 'Ground', 'Surface', 'Cobblestone road', 'Wet ground','Asphalt road', 'Race track', 'Floor', 'Concrete ground', 'Gravel surface', 'City road', 'City street', 'Lane', 'Expressway', 'Concrete surface', 'Path', 'Shop', 'Gray ground', 'Tarmac road', 'Tarmac surface', 'Sidewalk', 'Residential area', 'Country road', 'Coastal road', 'Dirt road', 'Country lane'],
    "Grassland":  ['Grassland', 'Lawn', 'Field', 'Green area', 'Vegetation', 'Shrubbery', 'Plant', 'Green belt', 'Green plants', 'Green vegetation', 'Landscaping', 'Green plants', 'Greenery'],
    "Exhibition hall": ['Exhibition hall', 'Showroom', 'Car sales showroom', 'Used car dealership', 'Used car sales lot', 'Car sales lot', 'Lighting', 'Exhibition', 'Hall', 'Car showroom', 'Light', 'Car exhibition', 'Car dealership', 'Expo', 'Car show', 'Street light', 'Car dealer', 'Second-hand car market',  'Car sales site', 'Studio', 'Display stand', 'Car expo', 'Display area', 'Store'],
    "Backdrop": ['Backdrop', 'Gray background', 'Billboard', 'Utility pole', 'Advertisement',  'Brick wall', 'Stone wall', 'Tile', 'Black background', 'Metal wall',  'Glass', 'Brick', 'Glass curtain wall', 'White background', 'Concrete wall', 'Stone brick', 'Red brick', 'Display board', 'Wall', 'Concrete', 'White wall', 'Blue', 'Graffiti', 'Red brick wall'],
    "Hill":['Mountain range', 'Hill', 'Mountain road', 'Mountainous terrain', 'Hillside', 'Hills', 'Cliff', 'Snow-capped mountain', 'Mountain'],
    "Sky": ['Sky', 'Blue sky', 'White clouds', 'Overcast', 'Dusk', 'Sunny day', 'Cloud layer', 'Clouds', 'Cloud', 'Night', 'Sunset', 'City skyline', 'Sunset'],
    "Fence": ["Fence","Guardrail","Barrier","Iron railing","Enclosure","Wooden fence","Wire fence"],
    "Rock": ["Rock"," Gravel"],
    "Beach": ["Beach","Coastline","Sand beach","Coast","Desert","Wasteland","Sand dunes"],
    "Lake":["Lake","Ocean","Sea","Lakeside","River","Seawave","Seawater","Water surface","Seashore","Island"],
    "Forest":  ["Forest","Wood"],
    "Snowfield": ["Snow accumulation","Snowfield","Winter"],
    "park":["park","garden,","flowers"],
   "autumn":["autumn","desolation","fallen leaves","fall"]
}


from diffusers import DiffusionPipeline
model_id = "yahoo-inc/photo-background-generation"
pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
pipeline = pipeline.to('cuda:0')
from PIL import Image, ImageOps
import requests
from io import BytesIO
from transparent_background import Remover
import torch
import cv2
import numpy as np
def resize_with_padding1(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding2(img, expected_size):
    original_size = img.size
    new_size = (original_size[0] * 2, original_size[1] * 2)
    img = img.resize(new_size, Image.BICUBIC)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


seed = 0

data_dict = {key: [] for key in keywords}
with open(file_path, 'r', encoding='utf-8') as f:
    count=0
    for line in f:
        values = line.strip().split(',')[1:7]
        leibie = line.strip().split(',')[0]
        folder_path = "data/Stanford_Cars/train/" + leibie
        image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]
        for i, key in enumerate(keywords):
            if i < len(values):
                if int(values[i]) < 15:
                    data_dict[key] = 15 - int(values[i])
                else:
                    data_dict[key] = 0
            else:
                data_dict[key] = None  

        for key, values in data_dict.items():
            print(key)
            print(values)
            for i in range(values):
                selected_image = random.choice(image_files)
                img = Image.open(selected_image)
                height,weight=img.size
                if img.size[0]<256 or img.size[1]<256:
                   img = resize_with_padding2(img, (512, 512))  
                else:
                    img = resize_with_padding1(img, (512, 512)) 
                # Load background detection model
                remover = Remover() # default setting
                remover = Remover(mode='base') # nightly release checkpoint

                # Get foreground mask
                fg_mask = remover.process(img, type='map') 
                seed = 13
                mask = ImageOps.invert(fg_mask)
                img = resize_with_padding1(img, (512, 512))
                generator = torch.Generator(device='cuda').manual_seed(seed) 
                select_words=random.choice(synonyms_dict[key])
                prompt ="a "+ leibie+" on the "+select_words+"."
                print(prompt)
                cond_scale = 1.0
                count+=1
                with torch.autocast("cuda"):
                    controlnet_image = pipeline(
                        prompt=prompt, image=img, mask_image=mask, control_image=mask, num_images_per_prompt=1, generator=generator, num_inference_steps=20, guess_mode=False, controlnet_conditioning_scale=cond_scale
                    ).images[0]
                outpath="/data/Stanford_Cars-backgroundMix-6-15/train/"+leibie+"/"+str(count)+".png"
                print(outpath)
                controlnet_image.save(outpath)
