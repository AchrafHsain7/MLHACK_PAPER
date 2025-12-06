import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torchvision
from torchvision import transforms
import os
import json
from PIL import Image
import re
import sklearn as sk
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CIFAR10_FSGM_DIR = f"/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-FSGM-eps8/train"
CIFAR10_PGD_DIR = f"/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-PGD-eps8/train"


def extract_images(path):
    image_files = [
    (f, int(m.group(1)))
    for f in os.listdir(path)
    if f.endswith((".jpg", ".png", ".jpeg"))
    if (m := re.search(r"idx=(\d+)", f))
    ]

    image_files = sorted(image_files, key=lambda x: x[1])
    print(image_files[:20])
    images_np = np.stack([
        np.array(Image.open(os.path.join(path, f[0])).convert('RGB'), dtype=np.float32) / 255.0
        for f in image_files
    ])
    return images_np



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
        # transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='../Datasets/', train=True, download=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    images_fsgm = np.load("/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-FSGM-eps8.npy")
    fsgm_labels = np.load("/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-FSGM-eps8/train/labels.npy")
    images_pgd = np.load("/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-PGD-eps8.npy")
    pgd_labels = np.load("/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-PGD-eps8/train/labels.npy")
    # np.save("/home/achraf/Research/MLHACK_PAPER/Datasets/CIFAR10-PGD-eps8.npy", images_pgd)

    print(images_pgd.shape)
    INDEX = 1

    fgsm_sim = []
    pgd_sim = []

    j=0
    for batch in tqdm(trainloader):
        for img in batch[0]:
            fgsm_sim.append(sk.metrics.pairwise.cosine_similarity(
                img.flatten().reshape(1,-1), images_fsgm[j].flatten().reshape(1,-1)  ))
            pgd_sim.append(sk.metrics.pairwise.cosine_similarity(
                img.flatten().reshape(1,-1), images_pgd[j].flatten().reshape(1,-1)  ))
            j+=1

    fgsm_sim = np.array(fgsm_sim).reshape(-1,1)
    pgd_sim = np.array(pgd_sim).reshape(-1,1)
    print(fgsm_sim.shape)

    # Have both data sim distribution and HOG distribution in same graph
    plt.hist(fgsm_sim, bins=50)
    plt.title("Cosine Similarity Distribution FGSM")
    plt.savefig("FGSM_cos_sim")
    plt.show()

    plt.hist(pgd_sim, bins=50)
    plt.title("Cosine Similarity Distribution PGD")
    plt.savefig("PGD_cos_sim")
    plt.show()


