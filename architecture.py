import torch
import torch.nn as nn
import torch.nn.functional as f
import config

# Større ændring vi skal have styr på, "Hvorfor vi bruger den padding vi bruger"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int=13, in_channels: int = 3): # Vi skal have ændret num_classes senere. in_channels = 3 er fint, da vi kører RGB farver.
        super().__init__()
        ### conv1 ###
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3) # Vi har 64 filtre, 7x7 kernel, 1 stride
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2) # Det første 2 tal er vores 2x2 kernel andet 2 tal er stride = 2

        ### Conv2 ###
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        ### Conv3 ###
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
            #Max pooling på kernel 2x2 og stride = 2
        )

        ### Conv4 ###
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            #Gentag ovenstående 3 yderligere gange
            nn.Conv2d(512, 256, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),


            nn.Conv2d(512, 512, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )

        ### Conv5 ###
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            #Gentag ovenstående 1 yderligere gang
            nn.Conv2d(1024, 512, kernel_size=1, stride=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
        )

        ### Conv6 ###
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected:
        D = config.B * 5 + config.C
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * config.S * config.S, 4096),
            # nn.dropout?
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, config.S * config.S * D)
        )
        self.S = config.S
        self.D = D

    def forward(self, x):
        # Aktiverer conv1 laget
        x = self.pool1(self.act1(self.conv1(x)))
        # Aktiverer conv2 laget
        x = self.pool2(self.act2(self.conv2(x)))

        # Aktiverer conv 3,4,5,6 lagende
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Fully connected 
        x = self.fc(x)

        # Reshape til YOLOv1 - format
        x = x.view(x.shape[0], self.S, self.S, self.D)
        return x
    

""" DETTE ER BARE EN LILLE TEST AF OUTPUT, for at sikre modellen kan køre værdierne"""
m = SimpleCNN()
x = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
with torch.no_grad():
    y = m(x)
print("Output shape:", y.shape)  # forventer: [1, 7, 7, B*5 + C]

print(sum(p.numel() for p in m.fc.parameters()), "param i FC")