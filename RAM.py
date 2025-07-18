import torchvision.transforms as T
from PIL import Image
import torch
from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference, get_transform
from PIL import Image

def load_tag_model(tag_model_weights="recognize-anything-plus-model/ram_plus_swin_large_14m.pth"):
    tag_model = ram_plus(
        pretrained=tag_model_weights,
        image_size=384,
        vit='swin_l',
    )
    tag_model.eval()
    tag_model.to("cuda")
    return tag_model

def get_tag(image, tag_model):
    """
    Get tags from the image using the tag model

    Args:
        image: Path to the image file
    """
    print("taging ......")
    transform = get_transform(image_size=384)
    # tag_model = tag_model.to("cuda")
    target = transform(Image.open(image)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        result = inference(target, tag_model)
    # print('tags:', result[0])
    tags = result[0]
    tags = [i for i in tags.split(' | ')]
    print("tags:", tags)
    return tags