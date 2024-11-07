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


class Generator_Mask_R_UNet(nn.Module):
    def __init__(self, n_channels = 1, maps = False):
        super(Generator_Mask_R_UNet, self).__init__()
        # e (encoder)
        self.down1 = UNetDown(n_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, dropout=0.5, normalize=False)
        # r (encoder)
        self.rep1 = UNetDown(n_channels, 64, normalize=False)
        self.rep2 = UNetDown(64, 128)
        self.rep3 = UNetDown(128, 256)
        self.rep4 = UNetDown(256, 512, dropout=0.5)
        self.rep5 = UNetDown(512, 512, dropout=0.5)
        self.rep6 = UNetDown(512, 512, dropout=0.5)
        self.rep7 = UNetDown(512, 512, dropout=0.5)
        self.rep8 = UNetDown(512, 512, dropout=0.5, normalize=False)
        self.mix = nn.Sequential(nn.Conv2d(1024,512, kernel_size=(3,3), padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        
        # d (decoder)
        self.up1 = UNetUp(512 , 512, dropout=0.5)
        self.up2 = UNetUp(1024+512, 512, dropout=0.5)
        self.up3 = UNetUp(1024+512, 512, dropout=0.5)
        self.up4 = UNetUp(1024+512, 512, dropout=0.5)
        self.up5 = UNetUp(1024+512, 256)
        self.up6 = UNetUp(512+256, 128)
        self.up7 = UNetUp(256+128, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128+64, n_channels, 4, padding=1),
            nn.Identity(),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )

    def forward(self, x, m, maps=None):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7) #=========> Look up here!
        #r (concatenar d con r en u )
        r1 = self.rep1(m)
        r2 = self.rep2(r1)
        r3 = self.rep3(r2)
        r4 = self.rep4(r3)
        r5 = self.rep5(r4)
        r6 = self.rep6(r5)
        r7 = self.rep7(r6)
        r8 = self.rep8(r7)
        mult  = self.mix(torch.cat([d8, r8],dim=1))
        u1 = self.up1(mult, torch.cat([d7, r7],dim=1))
        u2 = self.up2(u1, torch.cat([d6, r6],dim=1))
        u3 = self.up3(u2, torch.cat([d5, r5],dim=1))
        u4 = self.up4(u3, torch.cat([d4, r4],dim=1))
        u5 = self.up5(u4, torch.cat([d3, r3],dim=1))
        u6 = self.up6(u5, torch.cat([d2, r2],dim=1))
        u7 = self.up7(u6, torch.cat([d1, r1],dim=1))
        
        return self.final(u7)


##############################
#        Discriminator
##############################


class Mask_R_PatchGAN_Discriminator(nn.Module):
    def __init__(self, n_channels=3):
        super(Mask_R_PatchGAN_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(n_channels * 3, 64, normalization=False),
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

    net = Generator_Mask_R_UNet(n_channels = 1, maps = True)
    # summary(net, input_size=(1,256,256), device='cpu')

    dict_maps = OrderedDict()
    dict_maps['maps-256'] = torch.rand(1, 256, 64, 64)
    dict_maps['maps-512'] = torch.rand(1, 512, 32, 32)

    out = net(torch.rand(1, 1, 256, 256), maps = dict_maps)
    
    print('')


# torch.Size([1, 256, 64, 64])
# torch.Size([1, 512, 32, 32])

