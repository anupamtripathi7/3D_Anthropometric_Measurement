import cv2
import numpy as np

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

        self.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

def get_silhoutte(img):
    # Converting the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    silhoutte = cv2.Canny(gray, 100, 500)
    # Display the resulting frame
    cv2.imshow('Frame', silhoutte)
    cv2.waitKey(0)
    return silhoutte


def project_mesh_silhouette(mesh, angle):
    """
    Generate silhouette for projection of mesh at given angle
    Args:
        mesh (Mesh): SMPL mesh
        angle (int): Angle for projection

    Returns:
        silhouette
    """




if __name__ == "__main__":
    img = cv2.imread('human_0_270.jpg')
    get_silhoutte(img)