import torch
import torch.nn as nn

class Generator5x5(nn.Module):
    def __init__(self, latent_dim=32, num_classes=2, img_size=5):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, 16)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, img_size * img_size),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        class_vec = self.embed(y)
        
        x = torch.cat([z, class_vec], dim=1)
        
        img = self.net(x)
        
        return img.view(-1, 1, self.img_size, self.img_size)
    
class CNN5x5(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.fc(x)

        return x
    

class Generator16x16(nn.Module):
    def __init__(self, latent_dim=256, num_classes=2, img_size=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, 64)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        class_vec = self.embed(y)
        
        x = torch.cat([z, class_vec], dim=1)
        
        img = self.net(x)
        
        return img.view(-1, 1, self.img_size, self.img_size)


class CNN16x16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class Generator32x32(nn.Module):
    def __init__(self, latent_dim=512, num_classes=2, img_size=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes

        self.embed = nn.Embedding(num_classes, 128)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, img_size * img_size),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        class_vec = self.embed(y)
        
        x = torch.cat([z, class_vec], dim=1)
        
        img = self.net(x)
        
        return img.view(-1, 1, self.img_size, self.img_size)


class CNN32x32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x