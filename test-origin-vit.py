import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from utils.my_dataset import MyDataSet
# from vit_model import vit_base_patch16_224 as create_model
from utils.utils import read_data, evaluate
from vit_model.vit_model import  vit_base_patch16_224 as ViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_images_path, test_images_label = read_data("./robust/test-tomato-img")
#test_images_path, test_images_label = read_test_data("/media/xjw/doc/00-ubuntu-files/test-tomato")

data_transform = {
    "test": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

# 实例化训练数据集
test_dataset = MyDataSet(images_path=test_images_path,
                          images_class=test_images_label,
                          transform=data_transform["test"])
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=8,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=0,
                                           collate_fn=test_dataset.collate_fn)


model = ViT(num_classes=38)
model.to(device)
model_weight_path = "./robust/model-weight/vit-base-aug-tomato-8-2-from-scratch_weights.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


val_acc = evaluate(model=model,
                             data_loader=test_loader,
                             device=device,
                             epoch=1)

print(f"val_acc={val_acc}")


