import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
import re
import glob

path = "NOMO_preprocess/data"
batch_size = 1

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

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

    def __init__(self, folder="NOMO_preprocess/data"):
        measurements_path = os.path.join(folder, 'NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)')
        projections_path = os.path.join(folder, 'processed_data')

        for n, txt in enumerate(os.listdir(os.path.join(measurements_path, 'TC2_Female_Txt'))):
            with open(os.path.join(measurements_path, 'TC2_Female_Txt', txt)) as f:
                lines = f.read().strip().split('\n')[1:]
            file_n = txt[7:11]
            print(file_n)

            for angle in [0, 90, 180, 270]:
                img = cv2.imread(os.path.join(projections_path, 'female', 'human_{}_{}.jpg'.format()))


            if n == 2:
                break

        #
        # self.content = []
        #
        # for i, filename in enumerate(os.listdir(female_projections_path)):
        #     print(filename)
        #     if '_'+ str(i) +'_0' in filename:
        #         images[i, 0] = cv2.imread(os.path.join(female_projections_path, filename))
        #         print(images[i][0].shape)
        #     elif '_'+ str(i) +'_90' in filename:
        #         images[i, 1] = cv2.imread(os.path.join(female_projections_path, filename))
        #         print("a",images[i][0].shape)
        #     elif '_'+ str(i) +'_180' in filename:
        #         images[i, 2] = cv2.imread(os.path.join(female_projections_path, filename))
        #         print("a", images[i][0].shape)
        #     elif '_' + str(i) + '_270' in filename:
        #         images[i, 3] = cv2.imread(os.path.join(female_projections_path, filename))
        #         print("a", images[i][0].shape)
        #
        # print(images[0,1])
        # print(self.images[0].shape)
        #


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

    transformed_dataset = Nomo(path)

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    for i, sample in enumerate(dataloader):
        print(sample['images'].size())
        print(sample['content'])
