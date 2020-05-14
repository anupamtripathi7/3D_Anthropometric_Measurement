import os
import torch
from pytorch3d.io import load_obj, save_obj
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
import cv2


n_iter = 10

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SMPL mesh.
src_mesh = os.path.join('smpl.obj')

# Load the source and target images.
src_img = cv2.imread('src.jpg')
tar_img = cv2.imread('tar.jpg')

# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in src_mesh
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# The optimizer
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)


# Weight for the chamfer loss
w_chamfer = 1.0
# Weight for mesh edge loss
w_edge = 1.0
# Weight for mesh normal consistency
w_normal = 0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.1


chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


for i in range(n_iter):
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # Weighted sum of the losses
    loss = adversarial_loss(src_img, tar_img)

    # Print the losses
    print('Test Loss =  {}'.format(loss))

    # Optimization step
    loss.backward()
    optimizer.step()