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
    # print(len(adj_list), faces.shape)
    # print(adj_list)
    # print(adj_list[adj_list.keys()[1]])

