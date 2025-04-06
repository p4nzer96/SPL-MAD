import pandas as pd
import cv2

import torch

from torch.utils.data import Dataset

from pathlib import Path

from albumentations.pytorch import ToTensorV2
from albumentations import (Compose,
                            SmallestMaxSize,
                            RandomCrop,
                            HorizontalFlip,
                            ShiftScaleRotate,
                            ColorJitter,
                            Normalize,
                            CenterCrop)

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]


class TrainDataset(Dataset):
    """
    Train dataset for SPL-MAD

    :param csv_file: Path of data directory including csv files
    :type csv_file: Path
    :param input_shape: Model input shape
    :type input_shape: tuple[int]

    :return: Train dataset
    :rtype: Dataset
    """
    def __init__(self, csv_file: Path, input_shape: tuple[int] = (224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = Compose([
            SmallestMaxSize(max_size=input_shape[0]),
            RandomCrop(height=input_shape[0], width=input_shape[0]),
            HorizontalFlip(),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0.1, p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            Normalize(mean=PRE__MEAN, std=PRE__STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = self.composed_transformations(image=image)["image"]

        return {
            "images": image,
        }


    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = Compose([
            SmallestMaxSize(max_size=input_shape[0]),
            CenterCrop(height=input_shape[0], width=input_shape[0]),
            Normalize(mean=PRE__MEAN, std=PRE__STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]
        image = cv2.imread(img_path)
        label = 0 if label_str == "bonafide" else 1

        image = self.composed_transformations(image=image)["image"]

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "img_path": img_path,
        }

class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        if isinstance(csv_file, str):
            self.dataframe = pd.read_csv(csv_file)
        else:
            self.dataframe = csv_file
        self.composed_transformations = Compose([
            SmallestMaxSize(max_size=input_shape[0]),
            CenterCrop(height=input_shape[0], width=input_shape[0]),
            Normalize(PRE__MEAN, PRE__STD),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["Path"]
        label_str = self.dataframe.iloc[idx]["Label"]
        image = cv2.imread(img_path)
        label = 0 if label_str == 'bonafide' else 1

        image = self.composed_transformations(image=image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "img_path": img_path
        }
