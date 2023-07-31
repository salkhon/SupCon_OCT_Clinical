import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os


class CombinedDataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        if df.endswith(".xlsx"):
            self.df = pd.read_excel(df)
        else:
            self.df = pd.read_csv(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx, 0]
        if rel_path.startswith("/TREX DME"):
            rel_path = f"/TREX_DME{rel_path}"

        path = self.img_dir + rel_path
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)

        bcva = self.df.iloc[idx, 1]
        cst = self.df.iloc[idx, 2]
        eye = self.df.iloc[idx, 3]
        patient = self.df.iloc[idx, 4]

        return image, bcva, cst, eye, patient
