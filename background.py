from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "OpenGVLab/InternVL2-8B"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# set the max number of tiles in `max_num`
pixel_values = load_image('/examples/image.png', max_num=6).to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)



# single-round single-image conversation
question = 'Question="Based on this picture, I show a large military transport plane parked on the airport apron. From the details in the picture, this plane belongs to the US Air Force because there is a star shaped emblem on the tail, which is the logo of the US Air Force. Specifically, this plane is the C-17 Globemaster III transport plane produced by Lockheed Martin. The C-17 is a four engine jet military transport plane mainly used for strategic and tactical air transport missions. It has high payload capacity and long range, capable of transporting personnel, equipment, and materials. The aircraft in the picture is printed with the words"SAFARIR ", which is the logo of the Air Mobility Command of the United States Air Force. Air transportation services are responsible for global air transportation tasks, ensuring that troops and supplies can be quickly and safely delivered to the places where they are needed. In the background, the airport apron and some ground equipment can be seen, indicating that the aircraft may be undergoing loading and unloading operations or preparing for takeoff. Overall, this picture shows a US Air Force C-17 transport plane carrying out related missions at the airport The environmental keyword extracted from this sentence is: apron warehouse. Based on this picture, I show a plane taking off. The word "Tampa" is on the fuselage of the plane, indicating that it belongs to Tampa Airlines. From the appearance of the aircraft, it appears to be a Boeing 747, a large wide body passenger aircraft typically used for long-distance international flights. In the background, airport facilities can be seen, including runways, fences, and some buildings. The sky is clear and there are no obvious clouds, indicating good weather conditions suitable for flying. The text "Photo by Konstantin Von Wedelstadt" is located in the bottom left corner of the image, indicating that this photo was taken by Konstantin Von Wedelstadt. There is a watermark of "Airliners. net" in the bottom right corner, which is a website dedicated to collecting and sharing aerial photography works. Overall, this picture shows a scene of a Boeing 747 aircraft of Tampa Airlines taking off from the airport, The extracted environmental keyword is: sky. "'


import csv


input_file = 'result.csv'
output_file = 'background.csv'
with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    count=0
    for row in reader:
        column_data = row[-1]  
        add_question = question + "So according to the paragraph '{}', please provide the corresponding environmental element keywords (choose the most important and no more than three words to answer, without adjectives, pay attention to the environmental keywords)".format(column_data)

        response1, history1 = model.chat(tokenizer, pixel_values, add_question, generation_config, history=None, return_history=True)
        print(column_data)
        writer.writerow(row + [response1])
        print(count)
        count+=1
        print(response1)