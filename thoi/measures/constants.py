import torch

TWOPIE = torch.tensor(2 * torch.pi * torch.e)
LOGTWOPIE = torch.log(TWOPIE)
GAUS_ENTR_NORMAL = 0.5 * LOGTWOPIE