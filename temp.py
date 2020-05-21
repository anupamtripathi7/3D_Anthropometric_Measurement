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


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, idx=None):
        self.parent = parent
        self.idx = idx

        self.g = 0
        self.h = 0
        self.f = 0


def astar(adj_list, start_idx, end_idx, verts):
    print('In a*')
    # Create start and end node
    start_node = Node(None, start_idx)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end_idx)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = {start_node.idx: start_node}
    closed_list = {}
    counter = 0
    # Loop until you find the end
    while open_list:
        # if counter % 2 == 0:
        #     print(len(open_list), len(closed_list))

        # Get the current node
        current_index = list(open_list.keys())[0]
        for key, value in open_list.items():
            if value.f < open_list[current_index].f:
                current_index = key

        # Pop current off open list, add to closed list
        closed_list[current_index] = open_list[current_index]
        del open_list[current_index]

        # Found the goal
        if closed_list[current_index].idx == end_node.idx:

            path = []
            current = closed_list[current_index]
            while current is not None:
                path.append(current.idx)
                current = current.parent
            return path[::-1] # Return reversed path

        # Add the neighbours nodes to open list
        for neighbour in adj_list[closed_list[current_index].idx]:
            new_node = Node(closed_list[current_index], neighbour)
            new_node.g = closed_list[current_index].g + 0.01
            new_node.h = np.linalg.norm(verts[new_node.idx] - verts[end_node.idx])
            new_node.f = new_node.g + new_node.h
            if new_node.idx not in list(closed_list.keys()) and new_node.idx not in list(open_list.keys()):
                open_list[new_node.idx] = new_node
        counter += 1


def find_adj_list(verts, faces):
    print('In find_adj_list()')
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
    verts, faces_idx, _ = load_obj('data/male.obj')
    faces = faces_idx.verts_idx

    print(verts.size())
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)

    # for x in range(4370,4371):
    #     verts_rgb[0, x] = torch.tensor([1., 0., 0.])

    for n, v in enumerate(verts):
        # print(v)
        if v[1] >= -0.78 and v[1] <= -0.77 and v[0] >= -0.1 and v[0] <= 0:
            # print(v)
            # verts_rgb[0, n] = torch.tensor([1., 0., 0.])
            print(n)
    #
    #3474
    verts_rgb[0, 657] = torch.tensor([1., 0., 0.])
    # 629 and 1388 1679
    # print('Starting find_adj_list()')
    adj_list = find_adj_list(verts, faces)
    path1 = astar(adj_list, 5325, 1862, verts)
    # path4 = astar(adj_list, 3134, 1477, verts)
    path2 = astar(adj_list, 1679, 1388, verts)
    path3 = astar(adj_list, 1388, 629, verts)
    path = path1 + path2 + path3
    print(path1)

    #848, 964, 4167

    # 1784, 679, 4167
    # [679, 855, 920, 861, 858, 1769, 4344, 4345, 4404, 4341, 4167]
    # [1784, 5246, 5244, 6388, 6389, 4373, 4330, 4331, 4316, 4317, 4332, 4425, 4921, 4166, 4167]
    # [1784, 1781, 1780, 3122, 2928, 886, 845, 844, 831, 830, 846, 939, 1449, 678, 679]

    # for point in path1:
    #     verts_rgb[0, point] = torch.tensor([1., 0., 0.])

    # verts_rgb[0, 4386] = torch.tensor([0, 1, 0.])
    # verts_rgb[0, 964] = torch.tensor([0, 1, 0.])
    # verts_rgb[0, 848] = torch.tensor([0, 1, 0.])

    textures = Textures(verts_rgb=verts_rgb.to(device))

    R, T = look_at_view_transform(1.5, 0, 180, up=((0, 1, 0),), at=((0, 0, 0),))
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    raster_settings = RasterizationSettings(
        image_size=2048,
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


