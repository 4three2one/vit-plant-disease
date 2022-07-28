

import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


from visualizer import get_local
get_local.activate()
from vit_model import vit_base_patch16_224 as create_model
from PIL import Image, ImageDraw
import numpy as np

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "./tomato.JPG"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    get_local.clear()
    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    model_weight_path = "./vit-tomato-no-freeze-weights/epe-msa-bias1-vit-base-aug-tomato-8-2-from-scratch.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    cache=get_local.cache
    print(print_res)
    #plt.show()


def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows * 8.5, cols * 2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()


def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image
if __name__ == '__main__':
    main()