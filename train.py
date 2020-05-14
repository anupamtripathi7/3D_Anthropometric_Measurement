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
from model import Generator, Discriminator
import cv2


batch_size = 1
epochs = 50
d_lr = 1e-2
g_lr = 1e-2
beta = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_discriminator(d, real, generated, optimizer):
    d.zero_grad()

    # train on real images
    pred_real = d(real)
    loss_real = loss(pred_real, torch.ones(batch_size, 1))

    # train on generated images
    pred_generated = d(generated)
    loss_generated = loss(pred_generated, torch.zeros(batch_size, 1))

    if loss_generated + loss_real >= 0.01:
        loss_generated.backward()
        loss_real.backward()
        optimizer.step()


if __name__ == "__main__":

    src_mesh = os.path.join('smpl.obj')
    discriminator = Discriminator()
    generator = Generator()

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)

    # Load the source and target images.
    src_img = cv2.imread('src.jpg')
    tar_img = cv2.imread('tar.jpg')

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in src_mesh
    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

    for epoch in range(epochs):

        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)

        loss = 0
        print('Test Loss =  {}'.format(loss))
