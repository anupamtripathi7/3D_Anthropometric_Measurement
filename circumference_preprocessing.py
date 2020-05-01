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


if __name__ == "__main__":
    verts, faces_idx, _ = load_obj('Test/smpl_pytorch/human.obj')

    # print(verts.size())
    # print(faces_idx)
    # print(verts.size())
    # print(faces_idx)
    faces = faces_idx.verts_idx
    # print(faces)
    faces = faces.numpy()

    # print(faces.flatten())
    # print('b')
    adj_list = {idx: [] for idx in faces.flatten()}
   # flatten adj_list = dict.fromkeys(faces.flatten(), [])

    # for idx,face in enumerate(tqdm(faces)):
    #     for vertex in face:
    #         adj_list[vertex].append(face)
    #         # print(adj_list)
    #
    # print(adj_list)
    # print(faces[333])
    # print('b')
    # print(faces.shape)
    for idx, face in enumerate(tqdm(faces)):
        adj_list[face[0]].extend([face[1], face[2]])
        adj_list[face[1]].extend([face[0], face[2]])
        adj_list[face[2]].extend([face[0], face[1]])

        # print(idx, face, [(x, adj_list[x]) for x in range(0, 6)])

        adj_list[face[0]] = list(set(adj_list[face[0]]))
        adj_list[face[1]] = list(set(adj_list[face[1]]))
        adj_list[face[2]] = list(set(adj_list[face[2]]))

        # print(idx, face, [(x, adj_list[x]) for x in range(0, 6)])

    # print('a')
    # print('b',adj_list)
    print(len(adj_list), faces.shape)
    print(adj_list)
    # print(adj_list[adj_list.keys()[1]])

