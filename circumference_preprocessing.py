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
    verts, faces_idx, _ = load_obj(os.path.join(data_path, 'BioHuman_20200428.obj'))
    # print(verts.size())
    # print(faces_idx)
    faces = faces_idx.verts_idx
    # print(faces)
    faces = faces.numpy()
    print(faces)
    # print(faces.flatten())

    adj_list = dict.fromkeys(faces.flatten(), [])
    print(adj_list)
    print(faces.shape)

    # for idx,face in enumerate(tqdm(faces)):
    #     for vertex in face:
    #         adj_list[vertex].append(face)
    #         # print(adj_list)
    #
    # print(adj_list)

    face = faces[1]
    print(face[1])

    # for idx,face in enumerate(tqdm(faces)):
    #     print(face[1:2])
    #     for i, vertex in enumerate(face):
    #         adj_list[vertex].append(face[1:3])
    #
    #
            # print(adj_list)

    print(adj_list)

