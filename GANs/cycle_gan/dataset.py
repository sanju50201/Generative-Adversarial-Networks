import torch
from torch.utils.data import Dataset
import pandas as pd


class FinancialDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with financial data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values
        sample = torch.tensor(sample, dtype=torch.float32)  # Convert to tensor

        if self.transform:
            sample = self.transform(sample)

        return sample


# # Example usage:
# dataset = FinancialDataset(csv_file="../fraud.csv")
# print(len(dataset))
# print(dataset[0])  # Print the first sample
