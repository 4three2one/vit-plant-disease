import random
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    #read_split_data("/media/xjw/doc/00-ubuntu-files/Plant_leaf_diseases_augmentation")
    #read_split_data("/media/xjw/doc/00-ubuntu-files/archive/plantvillage dataset/color")
    #read_split_data("/media/xjw/doc/00-ubuntu-files/Plant_leaf_diseases_tomato_augmentation")
    #read_split_data("/media/xjw/doc/00-ubuntu-files/plantvillage_tomato")
    show()

def show():
    labels =['Background_without_leaves', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    orgin=[0,2127, 1000, 1908, 952, 1771, 1676, 1404, 5357, 373, 1591]
    aug=[1143, 2127, 1000, 1908, 1000, 1771, 1676, 1404, 5357, 1000, 1591]

    width = 0.35  # the width of the bars: can also be len(x) sequence
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    fig, ax = plt.subplots()

    ax.bar(labels, orgin, width,  label='原始数量')
    ax.bar(labels, (np.array(aug)-np.array(orgin)).tolist(), width,bottom=orgin,
           label='扩充数量')

    ax.set_ylabel('数量')
    ax.set_title('扩充后的数据集各个种类的分布')
    plt.xticks(rotation=60) # 旋转90度
    ax.legend()

    plt.show()

def read_split_data(root: str, val_rate: float = 0.1):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
    print(flower_class)
    print(every_class_num)


if __name__ == '__main__':
    main()