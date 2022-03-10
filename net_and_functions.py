from torch.utils.data import Dataset
import torch
import pandas as pd
from torchvision.transforms import RandomRotation

import os
from skimage import io
import numpy as np

class FishFinDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (): Path to the csv file with annotations.
            roostringt_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fish_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fish_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.fish_frame.iloc[idx, 0])
        image = torch.tensor(io.imread(img_name)/255)
        image = image.permute(2,0,1)
        combined_label = self.fish_frame.iloc[idx, 1]		#Scale to 0-3
        if combined_label == 1:
            label = 0
            label2 = 0
        elif combined_label == 2:
            label = 1
            label2 = 0
        elif combined_label == 3:
            label = 0
            label2 = 1
        elif combined_label == 4:
            label = 1
            label2 = 1

        #convert_strings = {'Has_fin':0, 'No_fin':1, 'Cannot_see':2}
        #convert_strings2 = {'No Fungi': 0, 'Slight Fungi':1, 'Severe Fungi':1}

        

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': torch.tensor([label]), 'label2': torch.tensor([label2])}

        return sample


