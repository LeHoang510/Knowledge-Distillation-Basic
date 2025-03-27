import os
import os.path as osp
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def visualize_sample(folder_path):
    folders = os.listdir(folder_path)
    list_imgs = []
    row = 2
    col = (len(folders)+row-1) // row
    
    for folder in folders:
        img = os.listdir(osp.join(folder_path, folder))[0]
        list_imgs.append((folder, osp.join(folder_path, folder, img)))

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

if __name__ == '__main__':
    visualize_sample(osp.join('dataset', 'weather-dataset', 'dataset'))