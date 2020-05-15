import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from NOMO import Nomo
from torch.utils.data import DataLoader
from utils import project_mesh_silhouette, Metadata
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
import os
import random


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x):
        pass


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_pass(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_pass(input1)
        output2 = self.forward_pass(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


epochs = 2
batch_size = 1
num_males = 179
num_females = 177
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smpl_mesh_path = "data/male.obj"
path = "NOMO_preprocess/data"

model = Discriminator()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number= 0

meta = Metadata()

mesh_male = [load_objs_as_meshes([os.path.join(meta.path, 'male.obj')], device=meta.device, load_textures=False)
                 for _ in range(meta.n_males)]
mesh_female = [load_objs_as_meshes([os.path.join(meta.path, 'female.obj')], device=meta.device, load_textures=False)
                   for _ in range(meta.n_females)]
mesh = {'male': mesh_male, 'female': mesh_female}




transformed_dataset = Nomo(folder=path)
dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, sample in enumerate(tqdm(dataloader)):
        for n, angle in enumerate([0, 90, 180, 270]):
            optimizer.zero_grad()
            projection = project_mesh_silhouette(mesh[sample['gender'][0]], angle)
            real_angle = angle + random.randint(-5, 5)
            real = project_mesh_silhouette(mesh[sample['gender'][0]], real_angle)
            fake = sample['images'][n]
            # img0, img1, label = data
            # img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            output1, output2 = model(projection, real)
            output3, output4 = model(projection, fake)

            loss_contrastive_pos = criterion(output1, output2, 0)
            loss_contrastive_neg = criterion(output3, output4, 1)
            loss_contrastive = loss_contrastive_neg + loss_contrastive_pos
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
# show_plot(counter,loss_history)
