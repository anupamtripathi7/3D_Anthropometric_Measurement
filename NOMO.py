import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch


path = "NOMO_preprocess/data"
batch_size = 1


class Nomo(Dataset):

    def __init__(self, folder="NOMO_preprocess/data"):
        measurements_path = os.path.join(folder, 'NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)')
        projections_path = os.path.join(folder, 'processed_data')

        for n, txt in enumerate(os.listdir(os.path.join(measurements_path, 'TC2_Female_Txt'))):
            with open(os.path.join(measurements_path, 'TC2_Female_Txt', txt)) as f:
                lines = f.read().strip().split('\n')[1:]

        projections = []
        for n, img in enumerate(os.listdir(os.path.join(projections_path, 'male'))):
            image = cv2.imread(os.path.join(projections_path, 'male', img), cv2.IMREAD_GRAYSCALE)
            projections.append(image)
            if n == 10:
                break
        self.projections = np.array(projections)
        print(np.array(projections).shape)

    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


if __name__ == "__main__":

    transformed_dataset = Nomo(folder=path)

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    # for x in dataloader:
    #     pass
