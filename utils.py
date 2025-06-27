import glob
from PIL import Image
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, train_flag=True, ratio=0.8, transform=None, target_transform=None):
        self.path = 'data/train/'
        self.class_label = {'apple': 0, 'cherry': 1, 'tomato': 2}
        self.data = {key: glob.glob(f'{self.path}{key}/*.jpg') for key in self.class_label.keys()}
        
        if train_flag:
            self.x_data, self.y_data = [], []
            for key in self.data.keys():
                self.x_data += self.data[key][:int(len(self.data[key])*ratio)]
                self.y_data += [self.class_label[key]] * len(self.data[key][:int(len(self.data[key])*ratio)])
        else:
            self.x_data, self.y_data = [], []
            for key in self.data.keys():
                self.x_data += self.data[key][int(len(self.data[key])*ratio):]
                self.y_data += [self.class_label[key]] * len(self.data[key][int(len(self.data[key])*ratio):])

        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.y_data)
    
    def __getitem__(self, idx):
        with Image.open(self.x_data[idx]) as image:
            label = self.y_data[idx]#.astype(np.int64)
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image.clone().contiguous(), label