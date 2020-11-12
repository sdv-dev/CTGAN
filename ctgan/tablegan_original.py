import numpy as np
import torch
from torch.nn import (
    BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, ReLU, Sequential, Sigmoid, Tanh, init)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from ctgan.transformer_tablegan import TableganTransformer
from torchsummary import summary

CATEGORICAL = "categorical"

# NOTE: This synthesizer was transferred from SDGym repository.
# It uses transformer_tablegan.py and normalizes continuous data using min-max method.
# To use this synthesizer, the input data must be in array format and column indices must be provided.

class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input_):
        return self.seq(input_)


class Classifier(Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, num_channels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    layers_G += [Tanh()]

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class TableganSynthesizerOriginal(object):
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500):

        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale

        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), epochs=300, model_summary=False):
        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= data.shape[1]:
                self.side = i
                break

        self.transformer = TableganTransformer(self.side)
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        data = self.transformer.transform(data)

        data = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim, self.num_channels)

        self.generator = Generator(self.transformer.meta, self.side, layers_G).to(self.device)
        discriminator = Discriminator(self.transformer.meta, self.side, layers_D).to(self.device)
        classifier = Classifier(
            self.transformer.meta, self.side, layers_C, self.device).to(self.device)

        if model_summary:
            print("*" * 100)
            print("GENERATOR")
            # in determine_layers, see side//2.
            summary(self.generator, (self.random_dim, self.side // 2, self.side // 2))
            print("*" * 100)

            print("DISCRIMINATOR")
            summary(discriminator, (1, self.side, self.side))
            print("*" * 100)

            print("CLASSIFIER")
            summary(classifier, (1, self.side, self.side))
            print("*" * 100)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)
        classifier.apply(weights_init)

        for i in range(epochs):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                optimizerG.zero_grad()
                y_fake = discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                if classifier.valid:
                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fake)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

                if((id_ + 1) % 50 == 0):
                    print("epoch", i + 1, "step", id_ + 1, loss_d, loss_g, loss_c)

    def sample(self, n):
        self.generator.eval()

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        return self.transformer.inverse_transform(data[:n])

