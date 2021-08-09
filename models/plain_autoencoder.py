import torch.nn as nn
import collections
import torch


# define model
class Autoencoder(nn.Module):
    def __init__(self, nbottleneck, batch_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ("encoder_linear1", nn.Linear(128 * 256, 100)),
                    ("encoder_relu1", nn.ReLU(True)),
                    ("encoder_linear2", nn.Linear(100, 49)), # 7x7
                    ("encoder_relu2", nn.ReLU(True)),
                    ("encoder_linear3", nn.Linear(49, 25)), # 5x5
                    ("encoder_relu3", nn.ReLU(True)),
                    ("encoder_linear4", nn.Linear(25, 9)), # 3x3
                    ("encoder_relu4", nn.ReLU(True)),
                    ("encoder_bottleneck", nn.Linear(9, nbottleneck)),
                ]
            )
        )

        self.decoder = nn.Sequential(
            collections.OrderedDict(
                collections.OrderedDict(
                [
                    ("decoder_linear1", nn.Linear(nbottleneck, 9)),
                    ("decoder_relu1", nn.ReLU(True)),
                    ("decoder_linear2", nn.Linear(9, 25)),
                    ("decoder_relu2", nn.ReLU(True)),
                    ("decoder_linear3", nn.Linear(25, 49)),
                    ("decoder_relu3", nn.ReLU(True)),
                    ("decoder_linear4", nn.Linear(49, 100)),
                    ("decoder_relu4", nn.ReLU(True)),
                    ("decoder_linear5", nn.Linear(100, 128 * 256)),
                    ("decoder_tanh", nn.Tanh()),
                ]
            )
        ))



    def forward(self, x):
        #print("input: ", x.size())
        #x = torch.squeeze(x)
        x = x.view(x.size(0), -1)
        #print("the reshaped x is of ", x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 128, 256)
        #print("output: ", x.shape)
        return x

    def generatefeatures(self, x):
        return self.encoder(x)
