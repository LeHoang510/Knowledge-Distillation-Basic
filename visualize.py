import os
import os.path as osp
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image


def visualize_sample(folder_path):
    folder = Path(folder_path)
    folders = [f for f in folder.iterdir() if f.is_dir()]
    list_imgs = []
    row = 2
    col = (len(folders)+row-1) // row
    
    for folder in folders:
        img = list(folder.glob('*.jpg'))[0]
        list_imgs.append((folder.name, img))

    fig, axs = plt.subplots(row, col, figsize=(16, 8))

    for i, (folder, img) in enumerate(list_imgs):
        img = Image.open(img)
        x, y = i // col, i % col
        axs[x, y].imshow(img)
        axs[x, y].set_title(folder)
        axs[x, y].axis('off')

    for i in range(len(folders), row*col):
        x, y = i // col, i % col
        axs[x, y].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_img(img: torch.Tensor):
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    visualize_sample(osp.join('dataset', 'weather-dataset', 'dataset'))