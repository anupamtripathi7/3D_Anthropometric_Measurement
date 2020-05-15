import cv2
import numpy as np
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


class Metadata:
    def __init__(self):
        self.batch_size = 1
        self.epochs = 50
        self.d_lr = 1e-2
        self.g_lr = 1e-2
        self.beta = 0.9
        self.inp_feature = 512 * 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smpl_mesh_path = "Test/smpl_pytorch/human.obj"
        self.raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])


def get_silhoutte(img):
    # Converting the image to grayscale.
    pass


def project_mesh_silhouette(mesh, angle):
    """
    Generate silhouette for projection of mesh at given angle
    Args:
        mesh (Mesh): SMPL mesh
        angle (int): Angle for projection

    Returns:
        silhouette
    """
    m = Metadata()
    R, T = look_at_view_transform(1.5, 0, angle, up=((0, 1, 0),), at=((0, 0.75, 0),))
    cameras = OpenGLPerspectiveCameras(device=m.device, R=R, T=T)
    raster_settings = m.raster_settings
    lights = m.lights
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(device=m.device, lights=lights)
    )
    verts_rgb = torch.ones((len(mesh.verts_list()), 1))[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(m.device))

    mesh.textures = textures
    mesh.textures._num_faces_per_mesh = mesh._num_faces_per_mesh.tolist()
    mesh.textures._num_verts_per_mesh = mesh._num_verts_per_mesh.tolist()

    image = renderer(mesh)
    image_cpy = image.clone()

    image_cpy = image.detach().cpu().numpy()[0, :, :, :-1]
    image_cpy = cv2.normalize(image_cpy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_cpy.astype(np.uint8)


    silhoutte = cv2.Canny(image_cpy, 100, 500)
    cv2.imshow('Frame', silhoutte)
    cv2.waitKey(0)
    # Display the resulting frame
    silhoutte = torch.Tensor(silhoutte)
    image[:,:,:] = silhoutte[:,:,:]


    return silhoutte

if __name__ == "__main__":
    m = Metadata
    img = cv2.imread('human_0_270.jpg')
    get_silhoutte(img)