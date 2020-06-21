import os
import torch
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import numpy as np
from model import Generator, Discriminator, ContrastiveLoss, Scale_Net
import cv2
from utils import project_mesh_silhouette, Metadata
from NOMO import Nomo
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance


# batch_size = 1
# epochs = 50
# d_lr = 1e-2
# g_lr = 1e-2
# beta = 0.9
# inp_feature = 512*512
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# smpl_mesh_path = "Test/smpl_pytorch/human.obj"
# path = "NOMO_preprocess/data"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def train_discriminator(d, c, projection, real, fake, optimizer):
#     output1, output2 = d(projection, real)
#     output3, output4 = d(projection, fake)
#     loss_contrastive_pos = c(output1, output2, 0)
#     loss_contrastive_neg = c(output3, output4, 1)
#     loss_contrastive = loss_contrastive_neg + loss_contrastive_pos
#     print('Test Loss =  {}'.format(loss_contrastive))
#     loss_contrastive.backward()
#     optimizer.step()
load = True
load_epoch = 39


if __name__ == "__main__":
    meta = Metadata()
    mesh_male = load_objs_as_meshes([os.path.join(meta.path, 'male.obj')], device=meta.device, load_textures=False)
    mesh_female = load_objs_as_meshes([os.path.join(meta.path, 'female.obj')], device=meta.device, load_textures=False)
    mesh = {'male': mesh_male, 'female': mesh_female}

    discriminator = Discriminator()
    # discriminator.load_state_dict(torch.load(os.path.join(meta.model_path, 'discriminator_21')))
    discriminator = discriminator.to(meta.device)
    for param in discriminator.parameters():
        param.requires_grad = False
    print(meta.device)
    print('loading data....')
    transformed_dataset = Nomo(folder=meta.path)
    dataloader = DataLoader(transformed_dataset, batch_size=meta.batch_size, shuffle=False)
    print('done')

    if load:
        deform_verts = pickle.load(open(os.path.join(meta.model_path, "deform_{}".format(str(load_epoch))), "rb"))
        discriminator.load_state_dict(torch.load(os.path.join(meta.model_path, "discriminator_{}".format(str(load_epoch)))))
    else:
        deform_verts = [torch.full(mesh['male'][0].verts_packed().shape, 0.0,
                               device=meta.device, requires_grad=True) for _ in range(len(dataloader))]

    criterion = ContrastiveLoss().to(meta.device)
    optimizer = torch.optim.Adam(list(discriminator.parameters()) + deform_verts, lr=meta.d_lr)

    model = Scale_Net()

    loss = torch.nn.MSELoss()

    for epoch in tqdm(range(meta.epochs)):
        epoch_loss = 0
        epoch_loss_supervised = 0
        for i, sample in enumerate(dataloader):
            for n, angle in enumerate([0, 90, 180, 270]):
                x_, y_ = [], []
                optimizer.zero_grad()
                new_mesh = mesh[sample['gender'][0]].offset_verts(deform_verts[i])
                projection = project_mesh_silhouette(new_mesh, angle)
                proj_img = projection.clone()
                # plt.imshow(proj_img.squeeze().detach().cpu().numpy())
                # plt.title('Epoch {} Angle {} Gender {}'.format(str(epoch), str(angle), sample['gender'][0]))
                # plt.show()
                real_angle = angle + random.randint(-5, 5)
                real = project_mesh_silhouette(new_mesh, real_angle)
                fake = sample['images'][0][n].unsqueeze(0).unsqueeze(0).to(meta.device)
                output1, output2 = discriminator(projection, real)
                loss_contrastive_pos = criterion(output1, output2, 0)
                output3, output4 = discriminator(projection, fake)
                loss_contrastive_neg = criterion(output3, output4, 1)
                loss_contrastive = loss_contrastive_neg + loss_contrastive_pos
                # if n == 3:
                #     loss_contrastive.backward()
                # else:
                #     loss_contrastive.backward(retain_graph=True)
                loss_contrastive.backward()
                optimizer.step()

                epoch_loss = loss_contrastive.detach()
                torch.cuda.empty_cache()

                optimizer.zero_grad()

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
                x_ = torch.cat(x_).unsqueeze(1)
                # print(x_.size())
                pred = model(x_)
                # print(pred.size())

                cost = loss(pred.squeeze(-1), torch.cat(y_).unsqueeze(1))
                cost.backward()
                optimizer.step()
                epoch_loss_supervised += cost.detach()


        if (epoch+1) % 10 == 0:
            pickle.dump(deform_verts, open("models/deform_{}".format(str(epoch)), "wb"))
            torch.save(discriminator.state_dict(), 'models/discriminator_{}'.format(str(epoch)))
        print("Epoch number {}\n Current loss {}\nSupervised loss {}\n".format(epoch, epoch_loss, epoch_loss_supervised))




