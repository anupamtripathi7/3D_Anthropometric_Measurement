import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch

path = "NOMO_preprocess/data"
batch_size = 2


class Nomo(Dataset):

    def __init__(self, folder="NOMO_preprocess/data"):
        measurements_path = os.path.join(folder, 'NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)')
        projections_path = os.path.join(folder, 'processed_data')

        data = self.get_data('male', measurements_path, projections_path)
        self.data = np.append(data, self.get_data('female', measurements_path, projections_path))

        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(self.data[idx]['images'])
        images = torch.tensor(self.data[idx]['images'])
        measurements = self.data[idx]['measurements']
        gender = self.data[idx]['gender']
        sample = {'images': images, 'measurements': measurements, 'gender': gender}

        return sample

    def get_data(self, gender, measurements_path, projections_path):
        data = []
        for n, txt in enumerate(os.listdir(os.path.join(measurements_path, 'TC2_' + gender[0].upper() + gender[1:] + '_Txt'))):
            with open(os.path.join(measurements_path, 'TC2_' + gender[0].upper() + gender[1:] + '_Txt', txt)) as f:
                lines = f.read().strip().split('\n')[1:]
            lines = list(map(lambda x: x.split()[1], lines))
            file_n = int(txt[7:11]) if gender == "female" else int(txt[5:9])

            images = []
            for angle in [0, 90, 180, 270]:
                img = cv2.imread(
                    os.path.join(projections_path, gender, 'human_{}_{}.jpg'.format(str(file_n), str(angle))))
                if img == None:
                    print(img, txt)
                images.append(img)
            images = np.array(images)
            data.append({'images': images, 'measurements': lines, 'gender': gender})
            if n == 2:
                break
        return np.array(data)


if __name__ == "__main__":

    transformed_dataset = Nomo(path)

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    for i, sample_1 in enumerate(dataloader):
        print(sample_1['images'].size())
        print(sample_1['measurements'])
        print(sample_1['gender'])
