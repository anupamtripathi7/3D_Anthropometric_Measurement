import torch
import torch.nn.functional as F

output1 = torch.tensor([[1, 2, 3, 4, 5]])
output2 = torch.tensor([[1, 2, 3, 4, 6]])
label = 1
margin = 2
euclidean_distance = F.pairwise_distance(output1, output2)
print(euclidean_distance, torch.pow(1/euclidean_distance, 2))
loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(1/euclidean_distance, 2))
print(loss_contrastive)