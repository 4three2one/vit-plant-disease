# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
import copy
from  tqdm import tqdm

# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out, noise # 这里也会返回噪声，注意返回值


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image
# 随机裁剪
def cutout(img, length=50,overimg=''):
    h=img.shape[0]
    w = img.shape[1]
    mask = np.ones((h, w))
    b, g, r = cv2.split(img)
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    mask[y1: y2, x1: x2] = 0
    rm = r * mask
    gm = g * mask
    bm = b * mask
    merged = cv2.merge([bm, gm, rm])
    ###img for overlap
    if overimg != '':
        over = cv2.imread(overimg)
        overresize = cv2.resize(over, (w, h), interpolation=cv2.INTER_CUBIC)
        mask2 = np.zeros((h, w))
        bo, go, ro = cv2.split(overresize)
        mask2[y1: y2, x1: x2] = 1
        b = bo * mask2
        g = go * mask2
        r = ro * mask2
        merged2 = cv2.merge([b, g, r])
        return merged + merged2
    return merged
def saltpepper():
    for rate in np.linspace(0.01, 0.1, 10):
        for dir in tqdm(os.listdir(file_dir)):
            category = os.path.basename(dir)
            for img_name in os.listdir(file_dir + dir):
                file_out = os.path.join(f'{file_dir[0:-1]}-SaltAndPepper',"{:.3f}".format(rate), category) + '/'
                if not os.path.exists(file_out):
                    os.makedirs(file_out)
                img_path = os.path.join(file_dir, category, img_name)
                img = cv2.imread(img_path)
                # cv2.imwrite(file_out + img_name,img)
                # 镜像
                # flipped_img = flip(img)
                # cv2.imwrite(file_dir + img_name[0:-4] + '_fli.jpg', flipped_img)

                # 增加噪声
                img_salt = SaltAndPepper(img, rate)
                cv2.imwrite(file_out + img_name[0:-4] + '_p.jpg', img_salt)
def blur():
    for rate in np.linspace(1,10,10):
        for dir in tqdm(os.listdir(file_dir)):
            category = os.path.basename(dir)
            for img_name in os.listdir(file_dir + dir):
                file_out = os.path.join(f'{file_dir[0:-1]}-blur', f'{rate}', category) + '/'
                if not os.path.exists(file_out):
                    os.makedirs(file_out)
                img_path = os.path.join(file_dir, category, img_name)
                img = cv2.imread(img_path)
                # 增加噪声
                blur = cv2.blur(img, (int(rate), int(rate)))
                # #      cv2.GaussianBlur(图像，卷积核，标准差）
                cv2.imwrite(file_out + img_name[0:-4] + '_blur.jpg', blur)
def GaussianBlur():
    for rate in np.arange(1, 15, 2):
        for dir in tqdm(os.listdir(file_dir)):
            category = os.path.basename(dir)
            for img_name in os.listdir(file_dir + dir):
                file_out = os.path.join(f'{file_dir[0:-1]}-GaussianBlur', f'{rate}', category) + '/'
                if not os.path.exists(file_out):
                    os.makedirs(file_out)
                img_path = os.path.join(file_dir, category, img_name)
                img = cv2.imread(img_path)
                # 增加噪声
                blur = cv2.GaussianBlur(img, (int(rate), int(rate)),0)
                # #      cv2.GaussianBlur(图像，卷积核，标准差）
                cv2.imwrite(file_out + img_name[0:-4] + '_gb.jpg', blur)
def occlusion():
    for rate in np.arange(60, 100, 5):
        for dir in tqdm(os.listdir(file_dir)):
            category = os.path.basename(dir)
            for img_name in os.listdir(file_dir + dir):
                file_out = os.path.join(f'{file_dir[0:-1]}-occlusion61', f'{rate}', category) + '/'
                if not os.path.exists(file_out):
                    os.makedirs(file_out)
                img_path = os.path.join(file_dir, category, img_name)
                img = cv2.imread(img_path)
                cut = cutout(img, rate)
                # #      cv2.GaussianBlur(图像，卷积核，标准差）
                cv2.imwrite(file_out + img_name[0:-4] + '_gblur.jpg', cut)
# 图片文件夹路径
file_dir = './test-tomato-img/'
blur()
# for img_name in os.listdir(file_dir):
#     img_path = file_dir + img_name
#     img = cv2.imread(img_path)
#     # cv2.imshow("1",img)
#     # cv2.waitKey(5000)
#     # 旋转
#     rotated_90 = rotate(img, 90)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_r90.jpg', rotated_90)
#     rotated_180 = rotate(img, 180)
#     cv2.imwrite(file_dir + img_name[0:-4] + '_r180.jpg', rotated_180)
            # img_gauss = addGaussianNoise(img, 0.3)
            # cv2.imwrite(file_out + img_name[0:-4] + '_noise.jpg', img_gauss)

            # 变亮、变暗
            # img_darker = darker(img)
            # cv2.imwrite(file_out + img_name[0:-4] + '_darker.jpg', img_darker)
            # img_brighter = brighter(img)
            # cv2.imwrite(file_out + img_name[0:-4] + '_brighter.jpg', img_brighter)
            #
            # blur = cv2.GaussianBlur(img, (7, 7), 1.5)
            # # #      cv2.GaussianBlur(图像，卷积核，标准差）
            # cv2.imwrite(file_out + img_name[0:-4] + '_blur.jpg', blur)

            # blur = cv2.blur(img, (7, 7))
            # # #      cv2.GaussianBlur(图像，卷积核，标准差）
            # cv2.imwrite(file_out + img_name[0:-4] + '_blur.jpg', blur)


