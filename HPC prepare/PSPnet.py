# -*- coding: utf-8 -*-
"""image-segmentation-from-scratch-in-pytorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qwEO3h_3I5jbyOwEgfvQHCITLVWM5fhF

# Image Segmentation From scratch using Pytorch

This is my first hands on with image segmentation and I tried to learn from existing pytorch notebooks.
One thing I imediately noticed is Using High level frameworks like catalyst is very convinient but For
learner like me who came here to learn it is difficult to know what is happening under the hood and
that's what we are here for,that is what we want to learn. so I wrote this kernel.
### It does all things from scartch so we can see what's happening

## Features of this kernel 
* Using vanila Unet Architecture
* Deterministic behaviour for reproducability
* K-fold cross validation is already Implemented (i.e. data spliting is done)
* loss function is also implmented for clearity
* Training loop is open to see exactly what is happening
* Processing output by removing mask that occur on black part of input image
* Drawing convex hull before optimizing thresholds
...

## Unet architecture
![Unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

Finally Huge thanks to **artgor, ryches, ratthachat, [repo1](https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/losses.py) ,[repo2](https://github.com/milesial/Pytorch-UNet) ** for their code

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


import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# ablumentations for easy image augmentation for input as well as output
import albumentations as albu
# from albumentations import torch as AT
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
        # self.masks = make_mask_all(self.df)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        # mask = self.masks[self.masks.Image_Label == image_name + "_" + classid]
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
                "understanding_cloud_organization/train_images_525/"
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
    masks = masks / 255
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


def visualize(image, mask, original_image=None, original_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}
    original_mask *= 225
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f"Mask {class_dict[i]}", fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title("Transformed image", fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(
                f"Transformed mask {class_dict[i]}", fontsize=fontsize
            )
    plt.savefig(save_path+"picture.png")


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
    


def plot_with_augmentation(image, mask, augment):
    """
    Wrapper for `visualize` function.
    """
    augmented = augment(image=image, mask=mask)
    image_flipped = augmented["image"]
    mask_flipped = augmented["mask"]
    visualize(image_flipped, mask_flipped, original_image=image, original_mask=mask)


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


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())

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

# sub = pd.read_csv(f"{path}/sample_submission.csv")
# sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
# sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
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

# split data
print(train.head())
id_mask_count = (
    train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"]
    .apply(lambda x: x.split("_")[0])
    .value_counts()
    .sort_index()
    .reset_index()
    .rename(columns={"Image_Label": "img_id", "count": "count"})
)
print(id_mask_count.info())
print(id_mask_count.head())
test_id_mask_count = id_mask_count[int(2*len(id_mask_count) / 3):]
id_mask_count = id_mask_count[:int(2*len(id_mask_count) / 3)]

ids = id_mask_count["img_id"].values
li = [
    [train_index, test_index]
    for train_index, test_index in StratifiedKFold(
        n_splits=N_FOLDS, random_state=SEED
    ,shuffle=True).split(ids, id_mask_count["count"])
]
train_ids, valid_ids = ids[li[MODEL_NO][0]], ids[li[MODEL_NO][1]]
test_ids = test_id_mask_count['img_id'].values
# test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values

print(f"training set   {train_ids[:5]}.. with length {len(train_ids)}")
print(f"validation set {valid_ids[:5]}.. with length {len(valid_ids)}")
print(f"testing set    {test_ids[:5]}.. with length {len(test_ids)}")

# define dataset and dataloader
num_workers = 0
bs = 4
train_dataset = CloudDataset(
    df=train,
    datatype="train",
    img_ids=train_ids,
    transforms=get_training_augmentation(),
)
valid_dataset = CloudDataset(
    df=train,
    datatype="valid",
    img_ids=valid_ids,
    transforms=get_validation_augmentation(),
)

train_loader = DataLoader(
    train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers
)
valid_loader = DataLoader(
    valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers
)

##modify
# sample_size = 500
# indices = random.sample(range(len(train_dataset)), sample_size)
# sampler = SubsetRandomSampler(indices)
# train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, num_workers=num_workers)

"""## Model Definition"""


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    

class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=3),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=6, stride=6),
            ConvBNReLU(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        branch1 = F.interpolate(self.branch1(x), size=(h, w), mode='bilinear', align_corners=True)
        branch2 = F.interpolate(self.branch2(x), size=(h, w), mode='bilinear', align_corners=True)
        branch3 = F.interpolate(self.branch3(x), size=(h, w), mode='bilinear', align_corners=True)
        branch4 = F.interpolate(self.branch4(x), size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat([x, branch1, branch2, branch3, branch4], dim=1)
    

class Classifier(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes, dropout):
        super().__init__()
        self.seg = nn.Sequential(
                ConvBNReLU(in_channels=in_channels, out_channels=mid_channels),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(in_channels=mid_channels, out_channels=num_classes, kernel_size=1, bias=True)
            )
          
    def forward(self, x):
        return self.seg(x)
    

class PSPNet(nn.Module):
    def __init__(self, num_classes, layers, is_training, pretrained):
        super().__init__()
        self.training = is_training
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        # print(resnet)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        self.ppm = PyramidPooling(in_channels=2048)
        self.seg = Classifier(in_channels=4096, mid_channels=512, num_classes=num_classes, dropout=0.1)
        self.aux = Classifier(in_channels=1024, mid_channels=256, num_classes=num_classes, dropout=0.1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.layer0(x)
        # print('# layer0 output shape:', x.shape)
        x = self.layer1(x)
        # print('# layer1 output shape:', x.shape)
        x = self.layer2(x)
        # print('# layer2 output shape:', x.shape)
        x_tmp = self.layer3(x)
        # print('# layer3 output shape:', x_tmp.shape)
        x = self.layer4(x_tmp)
        # print('# layer4 output shape:', x.shape)
        x = self.ppm(x)
        # print('# pyramid pooling module output shape:', x.shape)
        x = self.seg(x)
        # print('# seghead output shape:', x.shape)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # print('# seghead after upsampling shape:', x.shape)
        if self.training:
            aux_x = self.aux(x_tmp)
            # print('# Aux seghead output shape:', aux_x.shape)
            return x, aux_x
        return x


model = PSPNet(num_classes=4, layers=50, is_training=False, pretrained=False)
if train_on_gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

model # print Model

"""## Loss function definition"""

def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)

"""## RAdam Optimizer"""

import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom) 
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

criterion = BCEDiceLoss(eps=1.0, activation=None)
optimizer = RAdam(model.parameters(), lr = 0.005)
current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

"""## Training loop"""

# number of epochs to train the model
n_epochs = 32
train_loss_list = []
valid_loss_list = []
dice_score_list = []
lr_rate_list = []
valid_loss_min = np.Inf # track change in validation loss
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    dice_score = 0.0
    ###################
    # train the model #
    ###################
    print("Training Start")
    model.train()
    bar = tq(train_loader, postfix={"train_loss":0.0})
    for data, target in bar:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        #print(loss)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        bar.set_postfix(ordered_dict={"train_loss":loss.item()})
        # print(f"train epoch {epoch}: loss : {loss}")
    ######################    
    # validate the model #
    ######################
    model.eval()
    del data, target
    with torch.no_grad():
        bar = tq(valid_loader, postfix={"valid_loss":0.0, "dice_score":0.0})
        for data, target in bar:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
            dice_cof = dice_no_threshold(output.to(device), target.to(device)).item()
            dice_score +=  dice_cof * data.size(0)
            bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})  
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    dice_score = dice_score/len(valid_loader.dataset)
    print(f"train loss: {train_loss}, valid loss: {valid_loss}, dice score: {dice_score}")
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    dice_score_list.append(dice_score)
    lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])
    
    # print training/validation statistics 

    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifarpsp.pt')
        valid_loss_min = valid_loss
    
    scheduler.step(valid_loss)

"""## Ploting Metrics"""

new_model = PSPNet(num_classes=4, layers=50, is_training=False, pretrained=False)
# Get the current working directory
cwd = os.getcwd()

# List the contents of the directory
contents = os.listdir(cwd)

# Search for the model file
for file in contents:
    if file == 'model_cifarpsp.pt':
        print(f"Found model file at: {os.path.join(cwd, file)}")


plt.figure(figsize=(10,10))
plt.plot([i[0] for i in lr_rate_list])
plt.ylabel('learing rate during training', fontsize=22)
plt.savefig(save_path+"learning rate.png")
plt.show()

plt.figure(figsize=(10,10))
plt.plot(train_loss_list,  marker='o', label="Training Loss")
plt.plot(valid_loss_list,  marker='o', label="Validation Loss")
plt.ylabel('loss', fontsize=22)
plt.legend()
plt.savefig(save_path+"loss.png")
plt.show()

plt.figure(figsize=(10,10))
plt.plot(dice_score_list)
plt.ylabel('Dice score')
plt.savefig(save_path+"Dice Score.png")
plt.show()

# load best model
model.load_state_dict(torch.load('model_cifarpsp.pt'))
model.eval()

valid_masks = []
count = 0
tr = min(len(valid_ids)*4, 2000)
probabilities = np.zeros((tr, 350, 525), dtype = np.float32)
for data, target in tq(valid_loader):
    if train_on_gpu:
        data = data.cuda()
    target = target.cpu().detach().numpy()
    outpu = model(data).cpu().detach().numpy()
    for p in range(data.shape[0]):
        output, mask = outpu[p], target[p]
        for m in mask:
            valid_masks.append(resize_it(m))
        for probability in output:
            probabilities[count, :, :] = resize_it(probability)
            count += 1
        if count >= tr - 1:
            break
    if count >= tr - 1:
        break

"""## Grid Search for best Threshold"""

class_params = {}
for class_id in range(4):
    print(class_id)
    attempts = []
    for t in range(0, 100, 5):
        t /= 100
        for ms in [0, 100, 1200, 5000, 10000, 30000]:
            masks, d = [], []
            for i in range(class_id, len(probabilities), 4):
                probability = probabilities[i]
                predict, num_predict = post_process(probability, t, ms)
                masks.append(predict)
            for i, j in zip(masks, valid_masks[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))
            attempts.append((t, ms, np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]
    class_params[class_id] = (best_threshold, best_size)

del masks
del valid_masks
del probabilities
gc.collect()

attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
print(class_params)

attempts_df.groupby(['threshold'])['dice'].max()

attempts_df.groupby(['size'])['dice'].max()

attempts_df = attempts_df.sort_values('dice', ascending=False)
attempts_df.head(10)

sns.lineplot(x='threshold', y='dice', hue='size', data=attempts_df)
plt.title('Threshold and min size vs dice')
plt.savefig(save_path+"threshold.jpg")
best_threshold = attempts_df['threshold'].values[0]
best_size = attempts_df['size'].values[0]

for i, (data, target) in enumerate(valid_loader):
    if train_on_gpu:
        data = data.cuda()
    output = ((model(data))[0]).cpu().detach().numpy()
    image  = data[0].cpu().detach().numpy()
    mask   = target[0].cpu().detach().numpy()
    output = output.transpose(1 ,2, 0)
    image_vis = image.transpose(1, 2, 0)
    mask = mask.astype('uint8').transpose(1, 2, 0)
    pr_mask = np.zeros((350, 525, 4))
    for j in range(4):
        probability = resize_it(output[:, :, j])
        pr_mask[:, :, j], _ = post_process(probability,
                                           class_params[j][0],
                                           class_params[j][1])
    visualize_with_raw(image=image_vis, mask=pr_mask,
                      original_image=image_vis, original_mask=mask,
                      raw_image=image_vis, raw_mask=output,iter = i)
    if i >= 6:
        break

torch.cuda.empty_cache()
gc.collect()

test_dataset = CloudDataset(df=train,
                            datatype='test', 
                            img_ids=test_id_mask_count,
                            transforms=get_validation_augmentation())
test_loader = DataLoader(test_dataset, batch_size=4,
                         shuffle=False, num_workers=0)

del train_dataset, train_loader
del valid_dataset, valid_loader
gc.collect()

"""## Prepare Submission"""

# subm = pd.read_csv("understanding_cloud_organization/sample_submission.csv")
pathlist = ["understanding_cloud_organization/train_images_525/" + i for i in test_id_mask_count["img_id"]]

def get_black_mask(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (525,350))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0], np.uint8)
    upper = np.array([180, 255, 10], np.uint8)
    return (~ (cv2.inRange(hsv, lower, upper) > 250)).astype(int)

plt.imshow(get_black_mask(pathlist[120]))
plt.show()

encoded_pixels = []
image_id = 0
cou = 0
np_saved = 0
for data, target in tq(test_loader):
    if train_on_gpu:
        data = data.cuda()
    output = model(data)
    valid_loss += loss.item()*data.size(0)
    dice_cof = dice_no_threshold(output.to(device), target.to(device)).item()
    dice_score +=  dice_cof * data.size(0)
    bar.set_postfix(ordered_dict={"valid_loss":loss.item(), "dice_score":dice_cof})  
    # calculate average losses
dice_score = dice_score/len(valid_loader.dataset)
print(f"train loss: {train_loss}, valid loss: {valid_loss}, dice score: {dice_score}")
    # del data
    # for i, batch in enumerate(output):
    #     for probability in batch:
    #         probability = resize_it(probability.cpu().detach().numpy())
    #         predict, num_predict = post_process(probability,
    #                                             class_params[image_id % 4][0],
    #                                             class_params[image_id % 4][1])
    #         if num_predict == 0:
    #             encoded_pixels.append('')
    #         else:
    #             black_mask = get_black_mask(pathlist[cou])
    #             np_saved += np.sum(predict)
    #             predict = np.multiply(predict, black_mask)
    #             np_saved -= np.sum(predict)
    #             r = mask2rle(predict)
    #             encoded_pixels.append(r)
    #         cou += 1
    #         image_id += 1

# print(f"number of pixel saved {np_saved}")

# sub['EncodedPixels'] = encoded_pixels
# sub.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)

# """# Thank you for reading this do upvote if you like it"""