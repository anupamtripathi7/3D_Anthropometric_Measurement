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
import matplotlib.pyplot as plt

angle = 0
n = 0
meta = Metadata()

print('loading data....')
transformed_dataset = Nomo(folder=meta.path)
dataloader = DataLoader(transformed_dataset, batch_size=meta.batch_size, shuffle=True)
print('done')

for i, sample in enumerate(dataloader):
    sample = sample
    break

smpl_mesh = load_objs_as_meshes([os.path.join(meta.path, 'male.obj')])

discriminator = Discriminator()
discriminator = discriminator.to(meta.device)

criterion = ContrastiveLoss().to(meta.device)
optimizer = torch.optim.Adam(discriminator.parameters(), lr=meta.d_lr)

deform_verts = torch.full(smpl_mesh.verts_packed().shape, 0.0, device=meta.device, requires_grad=True)
#
smpl_mesh = smpl_mesh.offset_verts(deform_verts.clone())
projection = project_mesh_silhouette(smpl_mesh, angle).to(meta.device)
real_angle = angle + random.randint(-5, 5)
real = project_mesh_silhouette(smpl_mesh, real_angle).to(meta.device)
fake = sample['images'][0][n].unsqueeze(0).unsqueeze(0).to(meta.device)
# output1, output2 = discriminator(projection, real)

loss_contrastive_pos = criterion(projection, real, 0)
# output3, output4 = discriminator(projection, fake)
loss_contrastive_neg = criterion(projection, real, 1)
loss_contrastive = loss_contrastive_neg + loss_contrastive_pos
loss_contrastive.backward()
# optimizer.step()