import os
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader
)
from NOMO import Nomo
import pickle
from utils import Metadata
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import project_mesh_silhouette, Metadata
from tqdm import tqdm
from scipy.spatial import distance
from model import Scale_Net
from sklearn.metrics import r2_score
import numpy as np


meta = Metadata()
epochs = 25
# meta.device = torch.device("cpu")

data_path = "NOMO_preprocess/data"
model_path = "models"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
deform_verts = pickle.load(open(os.path.join(model_path, "deform_{}".format(str(39))), "rb"))
# print(deform_verts)
mesh_male = load_objs_as_meshes([os.path.join(meta.path, 'male.obj')], device=meta.device, load_textures=False)
mesh_female = load_objs_as_meshes([os.path.join(meta.path, 'female.obj')], device=meta.device, load_textures=False)

mesh = {'male': mesh_male, 'female': mesh_female}
transformed_dataset = Nomo(folder=meta.path)
dataloader = DataLoader(transformed_dataset, batch_size=meta.batch_size, shuffle=False)

model = Scale_Net()

loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=meta.d_lr)

for epoch in range(epochs):
    epoch_loss = 0
    for i, sample in enumerate(tqdm(dataloader)):
        for n, angle in enumerate([0, 90, 180, 270]):
            x_, y_ = [], []
            new_mesh = mesh[sample['gender'][0]].offset_verts(deform_verts[i])
            height, waist = 0, 0
            height_og, waist_og = 0, 0
            verts = new_mesh.verts_list()[0].detach()
            verts_og = mesh[sample['gender'][0]].verts_list()[0].detach()
            measurements = {_: 0.0 for _ in meta.measurements.keys()}
            measurements_og = {_: 0.0 for _ in meta.measurements.keys()}
            for m in meta.measurements.keys():
                for x in range(len(meta.measurements[m]['points']) - 1):
                    p1 = verts[meta.measurements[m]['points'][x]].cpu()
                    p2 = verts[meta.measurements[m]['points'][x + 1]].cpu()
                    measurements[m] += distance.euclidean(p1, p2)
                    p1 = verts_og[meta.measurements[m]['points'][x]].cpu()
                    p2 = verts_og[meta.measurements[m]['points'][x + 1]].cpu()
                    measurements_og[m] += distance.euclidean(p1, p2)
            # for x in range(len(meta.measurements['height'])-1):
            #     p1 = verts[meta.measurements['height'][x]].cpu()
            #     p2 = verts[meta.measurements['height'][x+1]].cpu()
            #     p11 = verts_og[meta.measurements['height'][x]].cpu()
            #     p22 = verts_og[meta.measurements['height'][x + 1]].cpu()
            #     height += distance.euclidean(p1, p2)
            #     height_og += distance.euclidean(p11, p22)
            # for x in range(len(meta.measurements['waist']) - 1):
            #     p1 = verts[meta.measurements['waist'][x]].cpu()
            #     p2 = verts[meta.measurements['waist'][x + 1]].cpu()
            #     p11 = verts_og[meta.measurements['waist'][x]].cpu()
            #     p22 = verts_og[meta.measurements['waist'][x + 1]].cpu()
            #     waist += distance.euclidean(p1, p2)
            #     waist_og += distance.euclidean(p11, p22)
            # if sample['gender'][0] == 'female':
            height_index = meta.measurements['waist']['ground_truth'][sample['gender'][0]]
            scale_factor = float(sample['measurements'][height_index][0].split('=')[1])/measurements['waist']
            for m in meta.measurements.keys():
                gt_index = meta.measurements[m]['ground_truth'][sample['gender'][0]]
                gt = float(sample['measurements'][gt_index][0].split('=')[1])
                x_.append(torch.FloatTensor([measurements_og[m]]))
                y_.append(torch.FloatTensor([measurements[m]]))
                # print('Mesh {}: {}\tDeformed: {}\tGround Truth: {}'.format(m, measurements_og[m]*scale_factor, measurements[m]*scale_factor, gt))
            # print('\n')
            optimizer.zero_grad()
            x_ = torch.cat(x_).unsqueeze(1)
            # print(x_.size())
            pred = model(x_)
            # print(pred.size())

            cost = loss(pred.squeeze(-1), torch.cat(y_).unsqueeze(1))
            cost.backward()
            optimizer.step()
            epoch_loss += cost.detach()
            print(r2_score(np.array(y_).reshape(-1, 1), pred.squeeze(-1).detach().cpu().numpy()))
    print(epoch_loss)
                # print('SMPL mesh waist: {}\t\tWaist after deform: {}\t\tTrouser waist{}\t\tNatural waist: {}'
                #       .format(waist_og * scale_factor, waist * scale_factor,
                #               float(sample['measurements'][4][0].split('=')[1]),
                #               float(sample['measurements'][41][0].split('=')[1])))
                # print(waist * scale_factor, float(sample['measurements'][4][0].split('=')[1]),
                #       float(sample['measurements'][41][0].split('=')[1]))
            # else:
            #     scale_factor = float(sample['measurements'][31][0].split('=')[1])/height
                # print(waist * scale_factor, float(sample['measurements'][9][0].split('=')[1]),
                #       float(sample['measurements'][10][0].split('=')[1]))
                # print('SMPL mesh waist: {}\t\tWaist after deform: {}\t\tTrouser waist{}\t\tNatural waist: {}'
                #       .format(waist_og * scale_factor, waist * scale_factor,
                #               float(sample['measurements'][9][0].split('=')[1]),
                #               float(sample['measurements'][10][0].split('=')[1])))
            # projection = project_mesh_silhouette(new_mesh, angle).to(meta.device)
            # proj_img = projection.clone()
            # plt.imshow(proj_img.squeeze().detach().cpu().numpy())
            # plt.title('Angle {} Gender {}'.format(str(angle), sample['gender'][0]))
            # plt.show()
