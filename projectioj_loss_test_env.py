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
from utils import project_mesh_silhouette, Metadata, project_mesh
from NOMO import Nomo
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch3d.structures import join_meshes_as_batch, Meshes, Textures


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

# smpl_mesh = load_objs_as_meshes([os.path.join(meta.path, 'male.obj')], device=meta.device)
verts, faces_idx, _ = load_obj(os.path.join(meta.path, 'male.obj'))
verts.requires_grad = True
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(meta.device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
smpl_mesh = Meshes(
    verts=[verts.to(meta.device)],
    faces=[faces.to(meta.device)],
    textures=textures
)

criterion = torch.nn.MSELoss()

projection = project_mesh(smpl_mesh, angle).to(meta.device)[0, :, :, 0]
plt.imshow(projection.detach().cpu().numpy())
plt.show()
fake = sample['images'][0][n].to(meta.device)
print(projection.type(), fake.type())
loss_contrastive_pos = criterion(projection, fake)
loss_contrastive_pos.backward()
# optimizer.step()