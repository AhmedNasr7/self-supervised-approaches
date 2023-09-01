import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class Custom_1D_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.data = []
        self.labels = []
        
        for label, directory in enumerate(self.directories):
            files = os.listdir(os.path.join(root_dir, directory))
            for file in files:
                file_path = os.path.join(root_dir, directory, file)
                data = np.load(file_path)
                self.data.append(data)
                self.labels.append(label)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label


class LitDataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.dataset = Custom_1D_Dataset(root_dir=self.root_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)


def test():
    # test_custom dataset

    dataset = Custom_1D_Dataset(root_dir='./data/')  # Specify the root directory where the subdirectories and files are located
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
    
    for batch_data, batch_labels in dataloader:
        # Do something with the batch data and labels
        print("batch: ", batch_data.shape, batch_labels.shape)
        print("label examples: ", batch_labels[0])

if __name__ == "__main__":
    test()
