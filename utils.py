import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
import os
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
from pytorch3d.structures import join_meshes_as_batch, Meshes, Textures
import torch
import time


class Metadata:
    def __init__(self):
        self.batch_size = 1
        self.epochs = 100
        self.d_lr = 1e-5
        self.g_lr = 1e-4
        self.beta = 0.9
        self.inp_feature = 512 * 512
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.smpl_mesh_path = "Test/smpl_pytorch/human.obj"
        self.path = "NOMO_preprocess/data"
        self.model_path = "models"
        self.raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.n_males = 179
        self.n_females = 177
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        self.measurements = {'height': {'points': [609, 3469], 'ground_truth': {'male': 31, 'female': 13}},
                             'waist': {'points': [679, 855, 920, 861, 858, 1769, 4344, 4345, 4404, 4341, 4167, 4166, 4921, 4425, 4332, 4317, 4316, 4331, 4330, 4373, 6389, 6388, 5244, 5246, 1784, 1781, 1780, 3122, 2928, 886, 845, 844, 831, 830, 846, 939, 1449, 678, 679], 'ground_truth': {'male': 9, 'female': 4}},
                             'shoulder': {'points': [5325, 4722, 4798, 5356, 5360, 5269, 4078, 4079, 4186, 4187, 6333, 3078, 2812, 697, 696, 589, 590, 1808, 1535, 1316, 1318, 1240, 1238, 1862], 'ground_truth': {'male': 11, 'female': 22}},
                             'outseam': {'points': [1802, 3319], 'ground_truth': {'male': 18, 'female': 1}},
                             'inseam': {'points': [1225, 3319], 'ground_truth': {'male': 19, 'female': 2}},
                             # 'hip_height': {'points': [1477, 3469],'ground_truth': {'male': 31, 'female': 13}},
                             'knee_height': {'points': [3474, 1178, 1175, 1102, 1103, 1077, 1076, 1116, 1121, 3191, 3323, 3208, 3469], 'ground_truth': {'male': 20, 'female': 6}},
                             'bust_circle': {'points': [1495, 1493, 2827, 1249, 3020, 3502, 6472, 4731, 4733, 4965, 4966, 6285, 4749, 4747, 4960, 6300, 6308, 6878, 4103, 4103, 4104, 4899, 6307, 4686, 4687, 1330, 1201, 1202, 2846, 1426, 615, 614, 614, 3480, 2847, 2839, 2840, 2851, 2841, 2825, 1495], 'ground_truth': {'male': 6, 'female': 18}},
                             'thigh_circle': {'points': [964, 909, 910, 1365, 907, 906, 957, 904, 848, 848, 849, 902, 851, 852, 898, 898, 899, 934, 935, 1453, 964], 'ground_truth': {'male': 15, 'female': 32}},
                             'calf': {'points': [1087, 1086, 1106, 1107, 1529, 1529, 1111, 1091, 1464, 1467, 1469, 1096, 1097, 1100, 1100, 1099, 1103, 1371, 1155, 1087], 'ground_truth': {'male': 17, 'female': 23}},
                             'hip_circle': {'points': [836, 838, 1230, 853, 854, 944, 850, 847, 1229, 1478, 1477, 1477, 3475, 914, 912, 1497, 3142, 3147, 3148, 4693, 4365, 4758, 4952, 6555, 6877, 4950, 4950, 4802, 4801, 6550, 6551, 6517, 6518, 6526, 6519, 6512, 4325, 4322, 1540, 836], 'ground_truth': {'male': 0, 'female': 5}},
                             'bicep': {'points': [629, 1678, 1716, 1679, 1679, 1314, 1379, 1378, 1394, 1393, 1389, 1388, 1388, 1234, 1231, 1386, 1384, 1737, 1398, 1395, 629], 'ground_truth': {'male': 23, 'female': 24}},
                             }


def get_silhoutte(img):
    # Converting the image to grayscale.
    pass


def project_mesh(mesh, angle):
    start = time.time()
    m = Metadata()
    R, T = look_at_view_transform(1.75, -45, angle, up=((0, 1, 0),), at=((0, -0.25, 0),))
    cameras = OpenGLPerspectiveCameras(device=m.device, R=R, T=T)
    raster_settings = m.raster_settings
    lights = m.lights
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(cameras=cameras, device=m.device, lights=lights)
    )
    verts = mesh.verts_list()[0]

    # faces = meshes.faces_list()[0]

    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    # verts_rgb = torch.ones((len(mesh.verts_list()[0]), 1))[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(m.device))

    mesh.textures = textures
    mesh.textures._num_faces_per_mesh = mesh._num_faces_per_mesh.tolist()
    mesh.textures._num_verts_per_mesh = mesh._num_verts_per_mesh.tolist()

    image = renderer(mesh)
    return image


def project_mesh_silhouette(mesh, angle):
    """
    Generate silhouette for projection of mesh at given angle
    Args:
        mesh (Mesh): SMPL mesh
        angle (int): Angle for projection

    Returns:
        silhouette
    """
    image = project_mesh(mesh, angle)

    silhoutte = image.data.clone()
    silhoutte = silhoutte.detach().cpu().numpy()[0, :, :, :-1]
    image_cpy = cv2.normalize(silhoutte, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    silhoutte = cv2.Canny(image_cpy, 100, 500)
    silhoutte = torch.tensor(silhoutte, dtype=torch.float32)
    image[0, :, :, 0].data = silhoutte.data
    image = image[:, :, :, :-3].permute(0, 3, 1, 2)

    return image


def load_mesh_from_obj(file, device):
    verts, faces_idx, _ = load_obj(file)
    faces = faces_idx.verts_idx

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))

    # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
    teapot_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    return teapot_mesh


if __name__ == "__main__":
    meta = Metadata()
    smpl_mesh = load_mesh_from_obj(os.path.join(meta.path, 'male.obj'), meta.device)
    plt.imshow(project_mesh_silhouette(smpl_mesh, 0).squeeze().detach().cpu().numpy())
    plt.show()