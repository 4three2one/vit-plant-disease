import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
# from vit_model import vit_base_patch16_224 as create_model
from utils import read_test_data, train_one_epoch, evaluate
from vit_pytorch.vit_for_small_dataset import ViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_images_path, test_images_label = read_test_data("/media/xjw/doc/00-ubuntu-files/test-tomato_augmentation")
#test_images_path, test_images_label = read_test_data("/media/xjw/doc/00-ubuntu-files/test-tomato")

data_transform = {
    "test": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(256),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

# 实例化训练数据集
test_dataset = MyDataSet(images_path=test_images_path,
                          images_class=test_images_label,
                          transform=data_transform["test"])



nw=3
print('Using {} dataloader workers every process'.format(nw))
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=8,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=nw,
                                           collate_fn=test_dataset.collate_fn)


model = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
model.to(device)
model_weight_path = "./sl-vit-tomato-lr002-weights/model-49.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


val_acc = evaluate(model=model,
                             data_loader=test_loader,
                             device=device,
                             )

print(f"val_acc={val_acc}")


