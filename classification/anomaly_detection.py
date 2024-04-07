from utils import CustomDataset 
from utils import Conv_AutoEncoder
from utils import transform

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


import os


def anomaly_detection(fp):
    fp = list(fp)
    images_dir = os.getcwd()
    data = CustomDataset(file_list = fp, root_dir = images_dir, transform=transform)

    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    ae = torch.load('validated_conv_autoencoder_50_epochs.pth')

    res = {}
    for img in dataloader:
        recon = ae(img)
        res['og'] = img
        res['recon'] = recon
    
    og = 0
    recon = 1

    original = res['og'][0][0]
    recon = res['recon'][0][0]
    mse = F.mse_loss(original, recon)

    return mse


anomaly_detection()