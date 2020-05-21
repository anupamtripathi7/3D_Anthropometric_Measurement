import os
import torch
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import numpy as np
from model import Generator, Discriminator, ContrastiveLoss
import cv2
from utils import project_mesh_silhouette, Metadata
from NOMO import Nomo
from torch.utils.data import DataLoader
from tqdm import tqdm
import random


# batch_size = 1
# epochs = 50
# d_lr = 1e-2
# g_lr = 1e-2
# beta = 0.9
# inp_feature = 512*512
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# smpl_mesh_path = "Test/smpl_pytorch/human.obj"
# path = "NOMO_preprocess/data"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_discriminator(d, c, projection, real, fake, optimizer):
    output1, output2 = d(projection, real)
    output3, output4 = d(projection, fake)
    loss_contrastive_pos = c(output1, output2, 0)
    loss_contrastive_neg = c(output3, output4, 1)
    loss_contrastive = loss_contrastive_neg + loss_contrastive_pos
    print('Test Loss =  {}'.format(loss_contrastive))
    loss_contrastive.backward()
    optimizer.step()


if __name__ == "__main__":
    print('a')

    meta = Metadata()
    mesh_male = [load_objs_as_meshes([os.path.join(meta.path, 'male.obj')], device=meta.device, load_textures=False)] * meta.n_males
    print(mesh_male)
    mesh_female = [load_objs_as_meshes([os.path.join(meta.path, 'female.obj')], device=meta.device, load_textures=False)] * meta.n_females
    mesh = {'male': mesh_male, 'female': mesh_female}

    discriminator = Discriminator()
    discriminator = discriminator.to(meta.device)

    criterion = ContrastiveLoss().to(meta.device)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=meta.d_lr)
    # g_optimizer = torch.optim.Adam(generator.parameters(), lr=meta.g_lr)

    print('loading data....')
    transformed_dataset = Nomo(folder=meta.path)
    dataloader = DataLoader(transformed_dataset, batch_size=meta.batch_size, shuffle=True)
    print('done')

    for epoch in range(meta.epochs):
        epoch_loss = 0
        for i, sample in enumerate(tqdm(dataloader)):
            deform_verts = torch.full(mesh['male'][i].verts_packed().shape, 0.0, device=meta.device,
                                      requires_grad=True)
            for n, angle in enumerate([0, 90, 180, 270]):
                print(n)

                optimizer.zero_grad()
                # mesh[sample['gender'][0]][i] = mesh[sample['gender'][0]][i].offset_verts(deform_verts)
                projection = project_mesh_silhouette(mesh[sample['gender'][0]][i], angle).to(meta.device)
                real_angle = angle + random.randint(-5, 5)
                real = project_mesh_silhouette(mesh[sample['gender'][0]][i], real_angle).to(meta.device)
                fake = sample['images'][0][n].unsqueeze(0).unsqueeze(0).to(meta.device)
                # train_discriminator(discriminator, criterion, projection, real, fake, d_optimizer)
                output1, output2 = discriminator(projection, real)

                loss_contrastive_pos = criterion(output1, output2, 0)
                output3, output4 = discriminator(projection, fake)
                loss_contrastive_neg = criterion(output3, output4, 1)
                loss_contrastive = loss_contrastive_neg + loss_contrastive_pos
                loss_contrastive.backward()
                optimizer.step()

                epoch_loss = loss_contrastive.detach()

        print("Epoch number {}\n Current loss {}\n".format(epoch, epoch_loss))



