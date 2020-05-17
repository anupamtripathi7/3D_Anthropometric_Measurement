# https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb


import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
    HardPhongShader
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = 'Test/test_pytorch3D/projection/data'

def find_adj_list(verts, faces):

    faces = faces.numpy()
    adj_list = {idx: [] for idx in faces.flatten()}

    for idx, face in enumerate(tqdm(faces)):
        adj_list[face[0]].extend([face[1], face[2]])
        adj_list[face[1]].extend([face[0], face[2]])
        adj_list[face[2]].extend([face[0], face[1]])

        adj_list[face[0]] = list(set(adj_list[face[0]]))
        adj_list[face[1]] = list(set(adj_list[face[1]]))
        adj_list[face[2]] = list(set(adj_list[face[2]]))

    return adj_list



if __name__ == "__main__":

    verts, faces, _ = load_obj('data/male.obj')
    adj_list = find_adj_list(verts, faces.verts_idx)
    # print(verts.size())
    # print(faces_idx)
    # print(verts.size())
    # print(faces_idx)


        # print(idx, face, [(x, adj_list[x]) for x in range(0, 6)])

    # print('a')
    # print('b',adj_list)
    print(len(adj_list), faces.shape)
    print(adj_list)
    # print(adj_list[adj_list.keys()[1]])


# Yeh
# https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb


import os
import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader
)
import cv2
from tqdm import tqdm
import numpy as np
# from astar import astar, find_adj_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '../../smpl_pytorch'


if __name__ == "__main__":
    verts, faces_idx, _ = load_obj(os.path.join(data_path, 'human.obj'))
    faces = faces_idx.verts_idx

    print(verts.size())
    #     verts = 100 * verts
    # print(verts)

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))

    R, T = look_at_view_transform(1.5, 0, 0, up=((0, 1, 0),), at=((0, 0, 0),))
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(device=device, lights=lights)
    )

    images = renderer(mesh)
    print(images.size())

    plt.imshow(images.detach().cpu().numpy()[0, :, :, :-1])
    plt.show()
    #     plt.savefig('fig.jpg')
    cv2.imwrite('fig.jpg', images.detach().cpu().numpy()[0, :, :, :-1])


