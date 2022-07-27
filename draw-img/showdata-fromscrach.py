import random
import matplotlib.pyplot as plt
import os
import numpy as np
import json

def main():
    with open("sl-vit-tomato-s8-d8.json", 'r', encoding='utf8') as fp:
        data=json.load(fp)
    acc=np.array(data)[:, 2]
    epoch=np.array(data)[:, 1]
    fig, ax = plt.subplots()
    ax.plot(epoch+1, acc)
    ax.set(xlabel='迭代次数', ylabel='训练准确率')
    plt.show()


if __name__ == '__main__':
    main()