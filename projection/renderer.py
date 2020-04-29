# https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/render_textured_meshes.ipynb


import os
import torch
import matplotlib.pyplot as plt
import cv2
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
    TexturedSoftPhongShader,
    HardPhongShader
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = 'data'


if __name__ == "__main__":
    verts, faces_idx, _ = load_obj(os.path.join(data_path, 'BioHuman_20200428.obj'))
    faces = faces_idx.verts_idx

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))

    R, T = look_at_view_transform(100, 10, 180)
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
        shader=HardPhongShader(device=device, lights=lights)
    )

    images = renderer(mesh)
    print(images.size())

    plt.imshow(images.detach().cpu().numpy()[0, :, :, :-1])
    plt.show()


