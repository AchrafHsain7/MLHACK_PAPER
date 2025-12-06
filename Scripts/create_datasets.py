import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn as sk
import skimage
import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import json
import re

def hogiffy(config, images, loader=False, path=None):
    hog_features = []
    labels = []
    if loader:
        for imgs, l in tqdm(images):
            for img in imgs:
                img = np.transpose(img, (1,2,0))
                gray = skimage.color.rgb2gray(img)
                feat = skimage.feature.hog(gray, orientations=config["ORIENTATIONS"], pixels_per_cell=(config["PXCELL"], config["PXCELL"]), 
                cells_per_block=(config["CELLBLOCK"], config["CELLBLOCK"]), block_norm='L2-Hys')
                hog_features.append(feat)
            # print(l)
            labels.extend([i.item() for i in l])
    else:
        for img in tqdm(images):
            gray = skimage.color.rgb2gray(img)
            feat = skimage.feature.hog(gray, orientations=config["ORIENTATIONS"], pixels_per_cell=(config["PXCELL"], config["PXCELL"]), 
                cells_per_block=(config["CELLBLOCK"], config["CELLBLOCK"]), block_norm='L2-Hys')
            hog_features.append(feat)
        labels = np.load(path+"/labels.npy")
    return np.array(hog_features), np.array(labels)


def extract_images(path):
    image_files = [
    (f, int(m.group(1)))
    for f in os.listdir(path)
    if f.endswith((".jpg", ".png", ".jpeg"))
    if (m := re.search(r"idx=(\d+)", f))
    ]

    image_files = sorted(image_files, key=lambda x: x[1])
    images_np = np.stack([
        np.array(Image.open(os.path.join(path, f[0])).convert('RGB'), dtype=np.float32) / 255.0
        for f in image_files
    ])
    return images_np










def create_triplet(config):
    ORIENTATIONS = config["ORIENTATIONS"]
    PXCELL = config["PXCELL"]
    CELLBLOCK = config["CELLBLOCK"]
    EPS = config["EPS"]
    CIFAR10_FSGM_DIR = f"/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-FSGM-eps{EPS}/train"
    CIFAR10_PGD_DIR = f"/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-PGD-eps{EPS}/train"

    # ORIGINAL CIFAR10 PROCESSING
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config["PXSIZE"]),
        #transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../Datasets/', train=True, download=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

    cifar10_orig_hog, labels = hogiffy(config, trainloader, True)
    print("ORIGINAL HOG data shape:", cifar10_orig_hog.shape)
    np.savez_compressed(f"../Datasets/cifar10-orig-hog-o{ORIENTATIONS}-c{PXCELL}-b{CELLBLOCK}", Y=labels, X=cifar10_orig_hog)
    del cifar10_orig_hog, labels, trainset

    # FGSM CIFAR10 PROCESSING
    images_np = extract_images(CIFAR10_FSGM_DIR)
    print("FSGM Images shape:", images_np.shape) 
    cifar10_fsgm_hog, labels = hogiffy(config, images_np, path=CIFAR10_FSGM_DIR)
    print("FSGM HOG data shape:",cifar10_fsgm_hog.shape)
    np.savez_compressed(f"../Datasets/cifar10-fsgm-hog-eps{EPS}-o{ORIENTATIONS}-c{PXCELL}-b{CELLBLOCK}", X=cifar10_fsgm_hog, Y=labels)
    del images_np, cifar10_fsgm_hog

    # FGSM CIFAR10 PROCESSING
    images_np = extract_images(CIFAR10_PGD_DIR)
    print("PGD Images shape:", images_np.shape) 
    cifar10_pgd_hog, labels = hogiffy(config, images_np, path=CIFAR10_PGD_DIR)
    print("PGD HOG data shape:",cifar10_pgd_hog.shape)
    np.savez_compressed(f"../Datasets/cifar10-pgd-hog-eps{EPS}-o{ORIENTATIONS}-c{PXCELL}-b{CELLBLOCK}", X=cifar10_pgd_hog, Y=labels)
    del images_np, cifar10_pgd_hog

    return [f"../Datasets/cifar10-orig-hog-o{ORIENTATIONS}-c{PXCELL}-b{CELLBLOCK}", f"../Datasets/cifar10-fsgm-hog-eps{EPS}-o{ORIENTATIONS}-c{PXCELL}-b{CELLBLOCK}", 
            f"../Datasets/cifar10-pgd-hog-eps{EPS}-o{ORIENTATIONS}-c{PXCELL}-b{CELLBLOCK}"]



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

if __name__ == "__main__":
    with open("/home/achraf/Research/MLHACK_PAPER/JSONS/configs.json", "r") as f:
        configs = json.load(f)

    infos = []
    for config in configs:
        infos.append(create_triplet(config))
    
    with open("/home/achraf/Research/MLHACK_PAPER/JSONS/files.json", "w") as f:
        json.dump(infos, f)
    



