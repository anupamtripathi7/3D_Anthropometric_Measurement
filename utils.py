import cv2
import numpy as np


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
    pass


if __name__ == "__main__":
    img = cv2.imread('human_0_270.jpg')
    get_silhoutte(img)