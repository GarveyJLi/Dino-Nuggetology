
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv_AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__() 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride = 3, padding = 1), # 32, 16, 67, 67
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride = 2, padding = 1),# 32, 32, 34, 34
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride = 2, padding = 1), # 32, 32, 17, 17
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 17 * 17, 3000),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3000, 32 * 17 * 17),
            nn.ReLU(),
            nn.Unflatten(dim = 1, unflattened_size=(32, 17, 17)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride = 2, padding = 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride = 3, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class CustomDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations to apply to your images
transform = transforms.Compose([
    # transforms.Resize((200, 200)), # not necessary, since images are already 200x200
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])