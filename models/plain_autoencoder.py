import torch.nn as nn

# define model
class Autoencoder(nn.Module):
    def __init__(self, nbottleneck):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear((1, 128 * 256), 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50, 25),
            nn.ReLU(True),
            nn.Linear(50, 25),
            nn.ReLU(True),
            nn.Linear(25, 12),
            nn.ReLU(True),
            nn.Linear(12, nbottleneck))

        self.decoder = nn.Sequential(
            nn.Linear(nbottleneck, 12),
            nn.ReLU(True),
            nn.Linear(12, 25),
            nn.ReLU(True),
            nn.Linear(25, 50),
            nn.ReLU(True),
            nn.Linear(50, 100),
            nn.ReLU(True),
            nn.Linear(100, 128 * 256),
            nn.Tanh())

    def forward(self, x):
        print("input: ", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def generatefeatures(self, x):
        return self.encoder(x)
