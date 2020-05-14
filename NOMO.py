import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch


path = "NOMO_preprocess/data"
batch_size = 1


# class Nomo1(Dataset):
#
#     def __init__(self, folder="NOMO_preprocess/data"):
#         measurements_path = os.path.join(folder, 'NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)')
#         projections_path = os.path.join(folder, 'processed_data')
#
#         for n, txt in enumerate(os.listdir(os.path.join(measurements_path, 'TC2_Female_Txt'))):
#             with open(os.path.join(measurements_path, 'TC2_Female_Txt', txt)) as f:
#                 lines = f.read().strip().split('\n')[1:]
#
#         projections = []
#         for n, img in enumerate(os.listdir(os.path.join(projections_path, 'male'))):
#             image = cv2.imread(os.path.join(projections_path, 'male', img), cv2.IMREAD_GRAYSCALE)
#             projections.append(image)
#             if n == 10:
#                 break
#         self.projections = np.array(projections)
#         print(np.array(projections).shape)
#
#     def __len__(self):
#         return len(self.projections)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

class Nomo(Dataset):

    def __init__(self, root="NOMO_preprocess/data", gender='female'):
        female_measurements_path = os.path.join(root, 'TC2_' + gender + '_Txt')
        female_projections_path = os.path.join(root, gender)
        self.images = []
        self.content = []
        for i, filename in enumerate(os.listdir(female_projections_path)):
            if i % 4 == 0:
                human = []
            human.append(cv2.imread(os.path.join(female_projections_path, filename)))
            if (i + 1) % 4 == 0:
                self.images.append(np.array(human))

        for i, filename in enumerate(os.listdir(female_measurements_path)):
            with open(os.path.join(female_measurements_path,filename)) as f:
                content = f.read()
            self.content.append(content.replace("MEASURE", "").split("\n")[1:])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = torch.tensor(self.images[idx])
        content = self.content[idx]
        sample = {'images': images, 'content': content}

        return sample


if __name__ == "__main__":

    transformed_dataset = Nomo(root=path)

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    for i, sample in enumerate(dataloader):
        print(sample['images'].size())
        print(sample['content'])
