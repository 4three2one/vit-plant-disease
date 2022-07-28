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
#from vit_model.vit_model import  vit_base_patch16_224 as model
from cnn_model.resnet import  resnet50 as model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root="./robust/test-tomato-img-output-SaltAndPepper"
dirs=[os.path.join(root, cla) for cla in os.listdir(root)]
acc=[]
for dir in dirs:
    test_images_path, test_images_label = read_data(dir)
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


    model = model(num_classes=38)
    model.to(device)
    model_weight_path = "./robust/model-weight/resnet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    val_acc = evaluate(model=model,
                                 data_loader=test_loader,
                                 device=device,
                                 epoch=1)

    acc.append(val_acc[1])
pass


