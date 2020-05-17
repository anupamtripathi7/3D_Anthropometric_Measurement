import torch
import cv2
import numpy as np
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from Test.smpl_pytorch.display_utils import display_model
import matplotlib.pyplot as plt
from pytorch3d.io import save_obj


if __name__ == '__main__':
    cuda = False
    batch_size = 1

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='male',
        model_root='/Users/anupamtripathi/PycharmProjects/3d_body_mesurement/Test/smpl_pytorch/smlppytorch/native/models')

    # Generate random pose and shape parameters
    pose_params = torch.ones(batch_size, 72) * 0
    shape_params = torch.ones(batch_size, 10)
    pose_params = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1, 0., 0., 1, 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).float()


    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()

    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

    # Draw output vertices and joints
    display_model(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath='image.png',
        show=True)
    print(verts.cpu().detach().size())
    save_obj('male.obj', verts.cpu().detach()[0], smpl_layer.th_faces)


# img = cv2.imread('/Users/anupamtripathi/PycharmProjects/3d_body_mesurement/Test/smplpytorch/image.png')
# print(img.shape)
# plt.imshow(img)
# plt.show()
