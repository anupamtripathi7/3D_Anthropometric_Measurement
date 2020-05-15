import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
import re
import glob

path = "NOMO_preprocess/data"
batch_size = 1


class Nomo(Dataset):

    def __init__(self, folder="NOMO_preprocess/data"):
        measurements_path = os.path.join(folder, 'NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)')
        projections_path = os.path.join(folder, 'processed_data')

        data = []
        for n, txt in enumerate(os.listdir(os.path.join(measurements_path, 'TC2_Female_Txt'))):
            with open(os.path.join(measurements_path, 'TC2_Female_Txt', txt)) as f:
                lines = f.read().strip().split('\n')[1:]
            lines = list(map(lambda x: x.split()[1], lines))
            file_n = int(txt[7:11])

            images = []
            for angle in [0, 90, 180, 270]:
                img = cv2.imread(os.path.join(projections_path, 'female', 'human_{}_{}.jpg'.format(str(file_n), str(angle))))
                images.append(img)
            images = np.array(images)
            data.append({'images': images, 'measurements': lines})
            if n == 2:
                break
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = torch.tensor(self.data[idx]['images'])
        content = self.data[idx]['measurements']
        sample = {'images': images, 'measurements': content}

        return sample


if __name__ == "__main__":

    transformed_dataset = Nomo(path)

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    for i, sample in enumerate(dataloader):
        print(sample['images'].size())
        print(sample['measurements'])
