import os
from pytorch3d.io import load_objs_as_meshes


def load_data(folder, device):
    objs = []
    for n, obj in enumerate(os.listdir(folder)):
        objs.append(os.path.join(folder, obj))
    meshes = load_objs_as_meshes(objs, device=device)
    return meshes