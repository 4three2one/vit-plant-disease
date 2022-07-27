import os
import json
import random
import shutil
from tqdm import tqdm
def cp_split_data(source: str, target: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(source), "dataset root: {} does not exist.".format(source)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(source) if os.path.isdir(os.path.join(source, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(source, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(source, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))
        target_cla_dir=f"./{target}/{cla}"
        if os.path.exists(target_cla_dir) is False:
            os.makedirs(target_cla_dir)

        for img_path in tqdm(images):
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                # val_images_path.append(img_path)
                # val_images_label.append(image_class)
                img_dir,img_name=os.path.split(img_path)
                if os.path.exists(os.path.join(target_cla_dir, img_name)):
                    print(f"文件{img_name}，已经存在，跳过拷贝... ")
                else:
                    shutil.copyfile(img_path, os.path.join(target_cla_dir, img_name) )

if __name__ == '__main__':
    cp_split_data(r"D:\dataset\plantvillage_aug_tomato","./test-tomato-img")