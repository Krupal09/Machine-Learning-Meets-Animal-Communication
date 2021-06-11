import torch.nn as nn
import collections

# define model
class Autoencoder(nn.Module):
    def __init__(self, nbottleneck):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ("encoder_linear1", nn.Linear(128 * 256, 100)),
                    ("encoder_relu1", nn.ReLU(True)),
                    ("encoder_linear2", nn.Linear(100, 50)),
                    ("encoder_relu2", nn.ReLU(True)),
                    ("encoder_linear3", nn.Linear(50, 25)),
                    ("encoder_relu3", nn.ReLU(True)),
                    ("encoder_linear4", nn.Linear(25, 12)),
                    ("encoder_relu4", nn.ReLU(True)),
                    ("encoder_bottleneck", nn.Linear(12, nbottleneck)),
                ]
            )
        )

        self.decoder = nn.Sequential(
            collections.OrderedDict(
                collections.OrderedDict(
                [
                    ("decoder_linear1", nn.Linear(nbottleneck, 12)),
                    ("decoder_relu1", nn.ReLU(True)),
                    ("decoder_linear2", nn.Linear(12, 25)),
                    ("decoder_relu1", nn.ReLU(True)),
                    ("decoder_linear3", nn.Linear(25, 50)),
                    ("decoder_relu1", nn.ReLU(True)),
                    ("decoder_linear4", nn.Linear(50, 100)),
                    ("decoder_relu1", nn.ReLU(True)),
                    ("decoder_linear5", nn.Linear(100, 128 * 256)),
                    ("decoder_tanh", nn.Tanh()),
                ]
            )
        ))



    def forward(self, x):
        #print("input: ", x.shape)
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 128, 256)
        #print("output: ", x.shape)
        return x

    def generatefeatures(self, x):
        return self.encoder(x)
