import torch.nn as nn
from models.residual_base import ResidualBase, get_block_sizes, get_block_type
from models.utils import get_padding
from models.residual_encoder import DefaultEncoderOpts

DefaultDecoderOpts = {
    "upsampling": "ConvTranspose2d", # MaxUnpool2d; (when maxpool is used, max unpooling; now only stride 2 is available)
    "output_channels": 1,
    "conv_kernel_size": 7,
    "input_channels": 512,
    "resnet_size": 18,
    "output_activation": "sigmoid",
    "output_stride": (2, 2),
}


class ResidualDecoder(ResidualBase):
    def __init__(self, opts: dict = DefaultDecoderOpts):
        super().__init__()
        self._opts = opts
        #self._layer_output = dict()
        self.cur_in_ch = opts["input_channels"]
        self.block_sizes = get_block_sizes(opts["resnet_size"])
        self.block_type = get_block_type(opts["resnet_size"]) # BasicBlock or Bottleneck

        self.hidden_layer = nn.Conv2d(
            4,
            out_channels=512,
            kernel_size=1,
            stride=(1, 1),
            padding=get_padding(1),
            bias=False,
        )

        # Upsample ResNet layers with transposed convolution
        # block_size in resnet 18 is [2, 2, 2, 2]

        """option 1 (used in killer whale): [256, 128, 64, 64]"""
        #self.layer1 = self.make_layer(
        #    self.block_type, 256, self.block_sizes[3], (2, 2), "upsample"
        #)  # 512 -> 256
        #self.layer2 = self.make_layer(
        #    self.block_type, 128, self.block_sizes[2], (2, 2), "upsample"
        #)  # 256 -> 128
        #self.layer3 = self.make_layer(
        #    self.block_type, 64, self.block_sizes[1], (2, 2), "upsample"
        #)  # 128 -> 64

        """option 2: [512, 256, 128, 64]"""
        self.layer1 = self.make_layer(
            self.block_type, 512, self.block_sizes[3], (2, 2), "upsample" # stride = (2,2) --> padding = (0,0)
        )  # 512 -> 512
        self.layer2 = self.make_layer(
            self.block_type, 256, self.block_sizes[2], (2, 2), "upsample"
        )  # 512 -> 256
        self.layer3 = self.make_layer(
            self.block_type, 128, self.block_sizes[1], (2, 2), "upsample"
        )  # 256 -> 128; what to do when
        if DefaultEncoderOpts["max_pool"] == 1:
            self.layer4 = self.make_layer(
                self.block_type, 64, self.block_sizes[0], (2, 2), "upsample"
            )  # 64 -> 64; use stride 2 instead of nn.MaxUnpool2d()

        self.conv_out = nn.ConvTranspose2d(
            in_channels=64 * self.block_type.expansion,
            out_channels=opts["output_channels"],
            kernel_size=opts["conv_kernel_size"],
            padding=get_padding(opts["conv_kernel_size"]),
            output_padding=get_padding(opts["output_stride"]),
            stride=opts["output_stride"],
            bias=False,
        )

        # TODO integrate also original encoder = decoder shapes in case of no max pooling by using stride2 in the last conv_out layer and stride (1,1) in the fourth residual layer
        #  (current variant is better because transposed convs with strid 2 in the last layer bring artifacts)

        if opts["output_activation"].lower() == "sigmoid":
            self.activation_out = nn.Sigmoid()
        elif opts["output_activation"].lower() == "relu":
            self.activation_out = nn.ReLU(inplace=True)
        elif opts["output_activation"].lower() == "tanh":
            self.activation_out = nn.Tanh()
        elif opts["output_activation"].lower() == "none":
            self.activation_out = lambda x: x
        else:
            raise NotImplementedError(
                "Unsupported output activation: {}".format(opts["output_activation"])
            )

    def forward(self, x):
        #print("In decoder, the shape of input is ", x.size())
        z = None
        if isinstance(x, tuple):
            # Extract latent code z and pass it through
            x, z = x
        # bring hidden layer back to (512, 16,8)
        #x = x.view(x.size(0), 16, 8)
        print("The shape of x going into decoder ", x.size())
        x = self.hidden_layer(x)
        print("In decoder, the shape of input into layer1 is ", x.size())
        # latent input data encoder
        #self._layer_output["input_layer"] = x
        # first residual layer start
        x = self.layer1(x)
        # first residual layer end and output
        #self._layer_output["residual_layer1"] = x
        # second residual layer start
        print("The shape of x going into layer2 is ", x.size())
        x = self.layer2(x)
        # second residual layer end and output
        #self._layer_output["residual_layer2"] = x
        # third residual layer start
        print("The shape of x going into layer3 is ", x.size())
        x = self.layer3(x)
        # third residual layer end and output
        #self._layer_output["residual_layer3"] = x
        # fourth residual layer start
        print("The shape of x going into layer4 is ", x.size())
        x = self.layer4(x)
        # fourth residual layer end and output
        #self._layer_output["residual_layer4"] = x
        print("The shape of x going into conv_out is ", x.size())
        x = self.conv_out(x)
        x = self.activation_out(x)
        # reconstructed spectrogram
        #self._layer_output["output_layer"] = x
        if z is not None:
            x = (x, z)
        print("The shape of x being returned is ", x.size())
        return x

    def model_opts(self):
        return self._opts

    def get_layer_output(self):
        return self._layer_output
