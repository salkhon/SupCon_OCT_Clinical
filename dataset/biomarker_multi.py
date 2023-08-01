import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os


class BiomarkerDatasetAttributes_MultiLabel(data.Dataset):
    def __init__(self, df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
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

        IRHRF_b1 = self.df.iloc[idx, 1]
        PAVF_b2 = self.df.iloc[idx, 2]
        FAVF_b3 = self.df.iloc[idx, 3]
        IRF_b4 = self.df.iloc[idx, 4]
        DRTME_b5 = self.df.iloc[idx, 5]
        VD_b6 = self.df.iloc[idx, 6]

        bio_tensor = torch.tensor([IRHRF_b1, PAVF_b2, FAVF_b3, IRF_b4, DRTME_b5, VD_b6])
        return image, bio_tensor
