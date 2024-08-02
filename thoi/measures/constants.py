import torch

TWOPIE = torch.tensor(2 * torch.pi * torch.e)
GAUS_ENTR_NORMAL = 0.5 * (torch.log(TWOPIE) + torch.log(torch.tensor(1.0)))