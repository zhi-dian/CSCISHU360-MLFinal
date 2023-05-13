# -*- coding: utf-8 -*-
"""image-segmentation-from-scratch-in-pytorch.ipynb

### **references**
1. https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools
2. https://www.kaggle.com/ryches/turbo-charging-andrew-s-pytorch
3. https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/losses.py
4. https://github.com/milesial/Pytorch-UNet
5. https://www.kaggle.com/ratthachat/cloud-convexhull-polygon-postprocessing-no-gpu

# Imports
"""

import os
import gc
import cv2
import time
import tqdm
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torchvision
from torchvision import models

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import albumentations as albu
plt.style.use('bmh')

# seeding function for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def resize_it(x):
    if x.shape != (350, 525):
        x = cv2.resize(x, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
    return x

# class_params = {0:(0.3,1200),1:(0.3,1200),2:(0.3,1200),3:(0.3,1200)}
# Dataset class
class CloudDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame = None,
        datatype: str = "train",
        img_ids: np.array = None,
        transforms=albu.Compose([albu.HorizontalFlip()]), #, AT.ToTensor()
    ):
        self.df = df
        self.data_folder = f"{img_paths}"
        self.img_ids = img_ids
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = np.transpose(augmented["image"], [2, 0, 1])
        mask = np.transpose(augmented["mask"], [2, 0, 1])
        return img, mask

    def __len__(self):
        return len(self.img_ids)

"""# Helper functions"""

# helper functions
class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']

def draw_convex_hull(mask, mode='convex'):
    
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        if mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255.

def get_img(x, folder: str = ""):
    """
    Return image based on image name and folder.
    """
    data_folder = f"{img_paths}/{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = "", shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if(mask_rle is np.nan):
        return np.zeros(shape[0] * shape[1], dtype=np.uint8)
    try:
        s = mask_rle.split()
    except:
        # print(mask_rle)
        return
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def make_mask(df: pd.DataFrame, image_name: str = "img.jpg", shape: tuple = (350, 525)):
    """
    Create mask based on df, image name and shape.
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    df = df[df["im_id"] == image_name]
    for idx, im_name in enumerate(df["im_id"].values):
        for classidx, classid in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
            mask = cv2.imread(
                path+"train_images_525/"
                + classid
                + im_name
            )
            if mask is None:
                continue
            # print(mask.sum())
            # print(mask)
            if mask[:, :, 0].shape != (350, 525):
                mask = cv2.resize(mask, (525, 350))
            masks[:, :, classidx] = mask[:, :, 0]
    # masks = masks / 255
    return masks

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype("float32")


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def visualize_with_raw(
    image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None, iter=None
):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}
    original_mask *= 225

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f"Raw predicted mask {class_dict[i]}", fontsize=fontsize)

    ax[2, 0].imshow(image)
    ax[2, 0].set_title("Transformed image", fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(
            f"Predicted mask with processing {class_dict[i]}", fontsize=fontsize
        )
    plt.savefig(save_path+"picture"+str(iter)+".png")
    

# sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def post_process(probability, threshold, min_size):
    """
    This is slightly different from other kernels as we draw convex hull here itself.
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = (cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1])
    mask = draw_convex_hull(mask.astype(np.uint8))
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=0.5,
            border_mode=0
        ),
        albu.GridDistortion(p=0.5),
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return albu.Compose(test_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(bool)
    img2 = np.asarray(img2).astype(bool)

    intersection = np.logical_and(img1, img2)

    return (2.0 * intersection.sum()+1e-7) / (img1.sum() + img2.sum()+1e-7)

def dice_no_threshold(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
):
    """
    Reference:
    https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/utils/criterion/dice.html
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()
    # else:
    #     outputs = (outputs > 0.5).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice

save_path = "datasaved/PSPnet/"
path = "understanding_cloud_organization/"
img_paths = "understanding_cloud_organization/train_image/"
train_on_gpu = torch.cuda.is_available()
SEED = 42
MODEL_NO = 0 # in K-fold
N_FOLDS = 10 # in K-fold
seed_everything(SEED)
torch.cuda.set_device(0)
print(os.listdir(path))
print(torch.cuda.device_count())
print(train_on_gpu)

"""## Make split in train test validation"""

train = pd.read_csv(f"{path}/train.csv")
train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

def make_img(df: pd.DataFrame,  shape: tuple = (350, 525)):
    """
    Create mask based on df, image name and shape.
    """
    # print(df.head())
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for idx, im_name in tq(enumerate(df["im_id"].values)):
        for classidx, classid in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
            enocded = df[df.Image_Label == im_name + "_" + classid]
            enocded = enocded.EncodedPixels
            encoded = enocded.iloc[0]
            mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
            if encoded is not np.nan:   
                mask = rle_decode(encoded)
                if mask[:, :].shape != (350, 525):
                    mask = cv2.resize(mask, (525, 350))
                masks[:, :, classidx] = mask
                # mask*=225
            cv2.imwrite(path+"train_images_525_withcolor/"+im_name[:-4]+classid+".jpg",mask)
    return masks

# make_img(train)

#analyze the masked pixel number
sz = {"Fish":0.,"Flower":0.,"Gravel":0.,"Sugar":0.}
for classidx,class_id in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
    for idx, im_name in tq(enumerate(train["im_id"].values)):
        # print(idx,im_name)
        img = cv2.imread(path+"train_images_525/"+class_id+im_name[:-4]+".jpg")
        # print(class_id,img)
        sz[class_id] += img.sum()/img.size
    sz[class_id]/=train["im_id"].values.size
print(sz)

sz = {"Fish":0.,"Flower":0.,"Gravel":0.,"Sugar":0.}
for classidx,class_id in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
    alll = train["im_id"].values.size
    for idx, im_name in tq(enumerate(train["im_id"].values)):
        # print(idx,im_name)
        img = cv2.imread(path+"train_images_525/"+class_id+im_name[:-4]+".jpg")
        # print(class_id,img)
        if img.sum() == 0:
            alll-=1
            continue
        sz[class_id] += img.sum()/img.size
    sz[class_id]/=alll
print(sz)

sz = {"Fish":0.,"Flower":0.,"Gravel":0.,"Sugar":0.}
for classidx,class_id in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
    alll = train["im_id"].values.size
    for idx, im_name in tq(enumerate(train["im_id"].values)):
        # print(idx,im_name)
        img = cv2.imread(path+"train_images_525/"+class_id+im_name[:-4]+".jpg")
        # print(class_id,img)
        if img.sum() == 0:
            alll-=1
            continue
        sz[class_id] += img.sum()/img.size
    sz[class_id]/=alll
print(sz)

all = {'Fish': 0.1482357691147128, 'Flower': 0.13884374937136745, 'Gravel': 0.15156360041311667, 'Sugar': 0.16502512223064408}
filt = {'Fish': 0.29561868950384657, 'Flower': 0.32559299535458935, 'Gravel': 0.2860060319466298, 'Sugar': 0.24399608848071236}
num = {'Fish': 0.5014424810674359, 'Flower': 0.4264334655607645, 'Gravel': 0.5299314821492968, 'Sugar': 0.6763433104940497}

for i in class_names:
    plt.bar(i,all[i])
plt.title("masks rate")
plt.savefig(save_path+"img_all.jpg")
plt.show()

for i in class_names:
    plt.bar(i,filt[i])
plt.title("masks size rate")
plt.savefig(save_path+"img_filt.jpg")
plt.show()

for i in class_names:
    plt.bar(i,filt[i])
plt.title("fig with obj rate")
plt.savefig(save_path+"img_num.jpg")
plt.show()

sz = {"Fish":0.,"Flower":0.,"Gravel":0.,"Sugar":0.}
for classidx,class_id in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
    alll = train["im_id"].values.size
    for idx, im_name in tq(enumerate(train["im_id"].values)):
        # print(idx,im_name)
        img = cv2.imread(path+"train_images_525/"+class_id+im_name[:-4]+".jpg")
        # print(class_id,img)
        if img.sum() == 0:
            alll-=1
            continue
        # sz[class_id] += img.sum()/img.size
    sz[class_id] = alll/train["im_id"].values.size
print(sz)