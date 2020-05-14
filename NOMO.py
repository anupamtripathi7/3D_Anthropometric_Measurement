import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


path = "NOMO_preprocess/data"
batch_size = 1


class Nomo(Dataset):

    def __init__(self, folder="NOMO_preprocess/data"):
        measurements_path = os.path.join(folder, 'NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)')
        projections_path = os.path.join(folder, 'processed_data')

        projections = np.empty((0, 512, 512, 3))
        for img in os.listdir(os.path.join(projections_path, 'male')):
            imgage = cv2.


if __name__ == "__main__":

    transformed_dataset = Nomo(folder=path)

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    for x in dataloader:
        pass
