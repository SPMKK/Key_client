import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
from ram.models import ram_plus
from ram import inference_ram as inference, get_transform
from PIL import Image
def load_tag_model(tag_model_weights="recognize-anything-plus-model/ram_plus_swin_large_14m.pth"):
    tag_model = ram_plus(
        pretrained=tag_model_weights,
        image_size=384,
        vit='swin_l',
    )
    return tag_model

def get_tag(image, tag_model):
    """
    Get tags from the image using the tag model

    Args:
        image: Path to the image file
    """
    print("taging ......")
    tag_model.eval()
    transform = get_transform(image_size=384)
    tag_model = tag_model.to("cuda")
    target = transform(Image.open(image)).unsqueeze(0).to("cuda")
    result = inference(target, tag_model)
    # print('tags:', result[0])
    tags = result[0]
    tags = [i for i in tags.split(' | ')]
    return tags

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model(model_name="OpenGVLab/InternVL3-2B"):
    ocr_model_path = model_name
    ocr_model = AutoModel.from_pretrained(ocr_model_path,
                                    torch_dtype=torch.bfloat16,
                                #   device_map="auto",
                                    load_in_8bit=True,
                                    trust_remote_code=True,
                                    use_flash_attn=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(ocr_model_path,
                                            trust_remote_code=True,
                                            use_fast=True)
    return ocr_model, tokenizer
def get_ocr(image, ocr_model, tokenizer):
    """
    Get OCR from the image using the OCR model
    
    Args:
        image: Path to the image file

    Returns:
        channel_name (str), main_news_text (str), thumbnail_text (str), time (str)
    """
    print("OCR.......")
    # Load and preprocess image
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()

    # Configure generation
    generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.01, top_p=0.9)

    # Define the question prompt
    question = '''
You are a structured OCR and information extraction model. Given a news broadcast image, extract key elements of the screen as structured data. Ignore any moving or scrolling text (tickers or subtitles). The banner may appear in any color and is typically used for concise summaries or headlines.

Extract the following fields (leave blank if not present):
**Channel Name:** The news channel name, typically visible as a logo or text in the top corners (e.g., HTV9, VTV, CNN, BBC, ANTV).

**Main News Text:**
- Paragraph-like or block text that conveys the main news content.
- Typically located in the center or main visual area of the screen (upper-middle or mid-screen).
- Includes large bold title text or speaker names and roles.
- Should not include branding, logos, or show names.
- If there is no informative news content, return main_news_text: null.

**Thumbnail Text:**
- A static overlay banner, regardless of its color.
- Positioned above the ticker, near the bottom, or on the side.
- Typically contains concise summaries, headlines, or topic teasers.
- Can appear in any color (red, blue, yellow, etc.).
- If the text is purely stylistic, logo-based, or branding-related (e.g., “60 giây” intro splash), return: "thumbnail_text": null.

**Time:** On-screen timestamp showing current broadcast time (e.g., 06:54:33).

**Return answer in this json format, no additional word:**
{
    "channel_name": "",
    "main_news_text": "",
    "thumbnail_text": "",
    "time": ""
}

Now extract the information from this image: 
<image>
'''

    # Generate response
    response = ocr_model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'Assistant: {response}')

    # Try parsing the response to extract the fields
    try:
        result = json.loads(response)
        channel_name = result.get("channel_name", "")
        main_news_text = result.get("main_news_text", "")
        thumbnail_text = result.get("thumbnail_text", "")
        time = result.get("time", "")
    except Exception as e:
        print(f"Error parsing OCR response: {e}")
        channel_name = ""
        main_news_text = ""
        thumbnail_text = ""
        time = ""

    return channel_name, main_news_text, thumbnail_text, time

# channel_name, main_news_text, thumbnail_text, time = get_ocr("/root/Keyframe_Extraction/data/L01_V001/frame_010.webp")
# print(channel_name, main_news_text, thumbnail_text, time, end = "\n")