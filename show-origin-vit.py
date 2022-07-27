from visualizer import get_local
get_local.activate()
from vit_model.vit_model import vit_base_patch16_224 as vit
from utils.visual_attention_utils import visualize_grid_to_grid_with_cls,visualize_heads
import torch
import torchvision.transforms as T
from PIL import Image
import json


image = Image.open('/home/xjw/PycharmProjects/vit-plant-disease/hotmap/pictures/tomato3.JPG')
imagenet_cls = json.load(open('/home/xjw/PycharmProjects/vit-plant-disease/hotmap/class_tomatos.json'))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize,
])

input_tensor = transforms(image).unsqueeze(0)
get_local.clear()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = vit(num_classes=10).to(device)
model_weight_path = "/home/xjw/PycharmProjects/vit-plant-disease/hotmap/origin-vit-tomato.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

with torch.no_grad():

    out = model(input_tensor.to(device))
print('Top1 prediction:')
print(imagenet_cls[str(out.argmax().item())])
cache = get_local.cache
print(list(cache.keys()))
attention_maps = cache['Attention.forward']
print(len(attention_maps))
attention_maps[0].shape
for i in range(0, 12):
    visualize_grid_to_grid_with_cls(attention_maps[2][0, i, :, :], 0, image)
# visualize_grid_to_grid(attention_maps[3][0,0,1:,1:], 100, image)
# visualize_head(attention_maps[7][0,1])
# visualize_heads(attention_maps[0], cols=4)
# visualize_head(attention_maps[7][0,1])
for i in range(0, 12):
    visualize_heads(attention_maps[i], cols=4)