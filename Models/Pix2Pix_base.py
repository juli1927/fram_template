import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != "PAConv":
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, n_channels = 3, maps = False):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(n_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512 , 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        # self.up6 = UNetUp(1024 if maps else 512, 128)
        # self.up7 = UNetUp(512  if maps else 256, 64)
        # self.lk1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        # self.lk2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, n_channels, 4, padding=1),
            nn.Identity(),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, x, maps=None):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        if maps: 
            # # u5 = torch.concat([u5, maps['maps-512']], dim=1)
            # # u5 = torch.sum(u5, maps['maps-512']*0.1)
            # u5 = u5 + maps['maps-512']*0.1

            maps['maps-512'] = self.lk1(maps['maps-512'])
            maps['maps-512'] = nn.InstanceNorm2d(maps['maps-512'].shape[1]) (maps['maps-512'])
            maps['maps-512'] = nn.LeakyReLU(0.2, inplace=True)(maps['maps-512'])
            u5 = u5 * (maps['maps-512'] * 0.5) #; print (u5.shape)
            # u5 = self.lk1(u5) #; print (u5.shape)
            u5 = nn.InstanceNorm2d(u5.shape[1]) (u5)
            u5 = nn.LeakyReLU(0.2, inplace=True)(u5)

        u6 = self.up6(u5, d2)
        
        if maps: 
            # # u6 = torch.concat([u6, maps['maps-256']], dim=1)
            # # u6 = torch.sum(u6, maps['maps-256']*0.1)
            # u6 = u6 + maps['maps-256']*0.1

            maps['maps-256'] = self.lk2(maps['maps-256'])
            maps['maps-256'] = nn.InstanceNorm2d(maps['maps-256'].shape[1]) (maps['maps-256'])
            maps['maps-256'] = nn.LeakyReLU(0.2, inplace=True)(maps['maps-256'])
            u6 = u6 * (maps['maps-256'] * 0.5) #; print (u6.shape)
            # u6 = self.lk2(u6) #; print (u6.shape)
            u6 = nn.InstanceNorm2d(u6.shape[1]) (u6)
            u6 = nn.LeakyReLU(0.2, inplace=True)(u6)
        
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class PatchGAN_Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(PatchGAN_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(n_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input) 


if __name__ == '__main__': 
    #
    from torchsummary import summary
    from torchvision.models._utils import IntermediateLayerGetter, OrderedDict

    net = GeneratorUNet(n_channels = 1, maps = True)
    # summary(net, input_size=(1,256,256), device='cpu')

    dict_maps = OrderedDict()
    dict_maps['maps-256'] = torch.rand(1, 256, 64, 64)
    dict_maps['maps-512'] = torch.rand(1, 512, 32, 32)

    out = net(torch.rand(1, 1, 256, 256), maps = dict_maps)
    
    print('')


# torch.Size([1, 256, 64, 64])
# torch.Size([1, 512, 32, 32])

