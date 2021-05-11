import torch.nn as nn

# define model
class vanillaAutoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(194 * 257, 24929),
            nn.ReLU(True),
            nn.Linear(24929, 12464),
            nn.ReLU(True),
            nn.Linear(12464, 6232),
            nn.ReLU(True),
            nn.Linear(6232, 3116),
            nn.ReLU(True),
            nn.Linear(3116, 1558),
            nn.ReLU(True),
            nn.Linear(1558, 779),
            nn.ReLU(True),
            nn.Linear(779, 390),
            nn.ReLU(True),
            nn.Linear(390, 195),
            nn.ReLU(True),
            nn.Linear(195, 98),
            nn.ReLU(True),
            nn.Linear(98, 49),
            nn.ReLU(True),
            nn.Linear(49, 25),
            nn.ReLU(True),
            nn.Linear(25, nbottleneck))

        self.decoder = nn.Sequential(
            nn.Linear(nbottleneck, 25),
            nn.ReLU(True),
            nn.Linear(25, 49),
            nn.ReLU(True),
            nn.Linear(49, 98),
            nn.ReLU(True),
            nn.Linear(98, 195),
            nn.ReLU(True),
            nn.Linear(195, 390),
            nn.ReLU(True),
            nn.Linear(390, 779),
            nn.ReLU(True),
            nn.Linear(779, 1558),
            nn.ReLU(True),
            nn.Linear(1558, 3116),
            nn.ReLU(True),
            nn.Linear(3116, 6232),
            nn.ReLU(True),
            nn.Linear(6232, 12464),
            nn.ReLU(True),
            nn.Linear(12464, 24929),
            nn.ReLU(True),
            nn.Linear(24929, 194 * 257),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def generatefeatures(self, x):
        return self.encoder(x)
