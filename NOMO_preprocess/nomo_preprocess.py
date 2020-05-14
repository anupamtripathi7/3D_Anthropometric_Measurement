import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from pytorch3d.structures import join_meshes_as_batch, Meshes, Textures
from NOMO_preprocess.utils import load_data
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



# print(os.listdir('.'))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data = "data/NOMO-3d-400-scans_and_tc2_measurements/nomo-scans(repetitions-removed)"


# if __name__ == "__main__":
    # mesh = load_data('female')
    # mesh = join_meshes_as_batch([mesh, load_data('male')])
    # print(len(mesh))
#

if __name__ == '__main__':
    meshes = load_data(os.path.join(data, 'male'), device=device)
    # meshes = join_meshes_as_batch([meshes, load_data(os.path.join(data, 'male'))])
    batch_verts = meshes.verts_list()
    batch_faces = meshes.faces_list()

    renderers = {}

    for j in [0, 90, 180, 270]:
        R, T = look_at_view_transform(1.5, 0, j, up=((0, 1, 0),), at=((0, 0.75, 0),))
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

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

        renderers[j] = renderer

    # print(renderers)

    for i, mesh in enumerate(tqdm(meshes)):
        for j in [0, 90, 180, 270]:

            verts = mesh.verts_list()[0]
            # faces = meshes.faces_list()[0]

            verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
            textures = Textures(verts_rgb=verts_rgb.to(device))


            mesh.textures = textures
            mesh.textures._num_faces_per_mesh = mesh._num_faces_per_mesh.tolist()
            mesh.textures._num_verts_per_mesh = mesh._num_verts_per_mesh.tolist()

            image = renderers[j](mesh)

            image = image.detach().cpu().numpy()[0, :, :, :-1]
            image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image.astype(np.uint8)

            cv2.imwrite(os.path.join('data/processed_data/Male2', 'human_{}_{}.jpg'.format(str(i), str(j))), image)
    # print(len(mesh))

