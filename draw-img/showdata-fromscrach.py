import random
import matplotlib.pyplot as plt
import os
import numpy as np
import json

def main():
    with open("./data/epe-msa-bias1-vit-base-aug-tomato-8-2-from-scratch.json", 'r', encoding='utf8') as fp:
        data_bias1_msa=json.load(fp)
    with open("./data/epe-msa-bias8-vit-aug-tomato-8-2-from-scratch.json", 'r', encoding='utf8') as fp:
        data_bias8_msa=json.load(fp)
    with open("./data/vit-base-aug-tomato-8-2-from-scratch.json", 'r', encoding='utf8') as fp:
        vit = json.load(fp)
    data_bias1_msa_acc=np.array(data_bias1_msa)[:, 2]
    data_bias8_msa_acc=np.array(data_bias8_msa)[:, 2]
    vit_acc=np.array(vit)[:, 2]
    epoch=np.array(data_bias1_msa)[:, 1]+1
    # coding=utf-8
    plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(epoch, data_bias1_msa_acc, marker='s', linestyle = '-.',color="blue", label="EPE_MMSA-ViT-bisa1", linewidth=1.2)
    plt.plot(epoch, data_bias8_msa_acc, marker='*', linestyle = '-.',color="green", label="EPE_MMSA-ViT-bisa8", linewidth=1.2)
    plt.plot(epoch, vit_acc, marker='^',linestyle = 'dotted', color="darkorange", label="ViT", linewidth=1.2)

    # group_labels = ['Top 0-5%', 'Top 5-10%', 'Top 10-20%', 'Top 20-50%', 'Top 50-70%', ' Top 70-100%']  # x轴刻度的标识
    plt.xticks(epoch,  fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("迭代次数", fontsize=13, fontweight='bold')
    plt.ylabel("训练准确率", fontsize=13, fontweight='bold')
    plt.xlim(3, 51)  # 设置x轴的范围
    plt.ylim(0.8, 1)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


if __name__ == '__main__':
    main()