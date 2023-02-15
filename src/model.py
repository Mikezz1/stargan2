import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        self.in_conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        self.out_conv = nn.Conv2d()
        self.downsampling = nn.Sequential(
            ResBlock(64, 128, nn.InstanceNorm2d()),
            nn.AvgPool2d(2),
            ResBlock(128, 256, nn.InstanceNorm2d()),
            nn.AvgPool2d(2),
            ResBlock(256, 512, nn.InstanceNorm2d()),
            nn.AvgPool2d(2),
            ResBlock(512, 512, nn.InstanceNorm2d()),
            nn.AvgPool2d(2),
            ResBlock(512, 512, nn.InstanceNorm2d()),
        )
        self.intermediate = nn.Sequential(
            ResBlock(512, 512, nn.InstanceNorm2d()),
            ResBlock(512, 512, nn.InstanceNorm2d()),
            AdaINResBlock(512, 512),
            AdaINResBlock(512, 512),
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            AdaINResBlock(512, 256),
            nn.Upsample(scale_factor=2),
            AdaINResBlock(256, 128),
            nn.Upsample(scale_factor=2),
            AdaINResBlock(128, 64),
            nn.Upsample(scale_factor=2),
            AdaINResBlock(64, 3),
        )

    def forward(self, x, s):
        x = self.in_conv(x)
        x = self.upsampling(self.intermediate(self.downsampling(x)), s)
        return self.out_conv(x)


class Discriminator(nn.Module):
    def __init__(self, D):
        self.in_conv = nn.Conv2d(3, 3, padding=1)
        self.downsampling = nn.Sequential(
            ResBlock(3, 64),
            nn.AvgPool2d(2),
            ResBlock(64, 128),
            nn.AvgPool2d(2),
            ResBlock(128, 256),
            nn.AvgPool2d(2),
            ResBlock(256, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
        )
        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(512, D * 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.downsampling(x)
        x = self.conv(x)
        return self.out(x)


class MappingNetwork(nn.Module):
    """Generates style codes \mathcal{s} for every domain \mathcal{y}
    given some latent code \mathcal{z}. Used during training for generating
    images with random a) domain and b) style. Trained with generator via
    adversarial objective and style diversificaton loss
    """

    def __init__(self, K):
        self.backbone = nn.Sequential(
            *list(self.get_fc_block(512, 512) for i in range(4))
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    self.get_fc_block(512, 512),
                    self.get_fc_block(512, 512),
                    self.get_fc_block(512, 512),
                    self.get_fc_block(512, 64, activation=False),
                )
                for j in range(K)
            ]
        )

    def forward(self, z):
        z = self.backbone(z)
        outputs = []
        for head in self.heads:
            outputs.append(head(z))

        return outputs

    def get_fc_block(self, in_ch, out_ch, activation=True):
        return (
            nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU())
            if activation
            else nn.Linear(in_ch, out_ch)
        )


class StyleEncoder(nn.Module):
    """Maps picture to style code. Trained with style reconstruction loss,
    hwich is L1 distance between fixed style_code and style encoder prediction
    given generated image.
    """

    def __init__(self, D):
        self.in_conv = nn.Conv2d(3, 3, padding=1)
        self.downsampling = nn.Sequential(
            ResBlock(3, 64),
            nn.AvgPool2d(2),
            ResBlock(64, 128),
            nn.AvgPool2d(2),
            ResBlock(128, 256),
            nn.AvgPool2d(2),
            ResBlock(256, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
        )
        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.out = nn.Linear(512, D * 64)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.downsampling(x)
        x = self.conv(x)
        return self.out(x)


class ResBlock(nn.Module):
    """
    Building block for Generator, Discriminator and Style Encoder.
    Should specify either pooling or upsampling

    Args:
        norm (nn.Module): Instance norm or Adaptive Instance norm
        act (nn.Module): Activation function
        pre_resampler (nn.Module): ConvTranspoce or Upsample for upsampling blocks
        post_resampler (nn.Module): pooling layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm=DummyResampler(),
        act=nn.Relu(),
        # upsampling=DummyResampler(),
        # pooling=DummyResampler(),
    ):
        self.norm = norm
        # self.upsampling = upsampling
        # self.pooling = pooling
        self.act = act
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.out_channels < self.in_channels:
            self.proj_conv = nn.Conv2d(
                in_channels, out_channels, 1, padding=0, bias=False
            )

        self.layers = nn.Sequential(
            self.norm,
            self.act,
            # self.upsampling,
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            # self.pooling,
            self.norm,
            self.act,
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)

    def shortcut(self, x):
        if self.out_channels > self.in_channels:
            return self.proj_conv(x)
        if self.out_channels > self.in_channels:
            return F.upsample(x, scale_factor=2, mode="bilinear")
        return x


class AdaINResBlock(nn.Module):
    """
    Building block for Generator

    Args:
        norm (nn.Module): Instance norm or Adaptive Instance norm
        act (nn.Module): Activation function
        pre_resampler (nn.Module): ConvTranspoce or Upsample for upsampling blocks
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        act=nn.Relu(),
        upsampling=DummyResampler(),
    ):
        # self.upsampling = upsampling
        # self.pooling = pooling
        self.act = act
        self.style_dim = 64

        self.layers = nn.Sequential(
            AdaIN(in_channels, self.style_dim),
            self.act,
            # self.upsampling,
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            # self.pooling,
            AdaIN(in_channels, self.style_dim),
            self.act,
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x, s):
        resid = x
        for layer in self.layers:
            x = layer(x) if not isinstance(layer, AdaIN) else layer(x, s)
        return self.shortcut(resid) + x

    def shortcut(self, x):
        if self.out_channels > self.in_channels:
            return self.proj_conv(x)
        if self.out_channels > self.in_channels:
            return F.upsample(x, scale_factor=2, mode="bilinear")
        return x


class AdaIN(nn.Module):
    """Like in style gan"""

    def __init__(self, n_fmaps, style_dim):
        self.mu_projector = nn.Linear(style_dim, n_fmaps)
        self.sigma_projector = nn.Linear(style_dim, n_fmaps)

    def forward(self, x, s):

        mu_style = self.mu_projector(s)
        sigma_projector = self.mu_projector(s)
        # x ~ (B, C, H, W), mu/std ~ (B, C)
        mu_style = mu_style.unsqueeze(2).unsqueeze(3)
        sigma_projector = sigma_projector.unsqueeze(2).unsqueeze(3)

        x = F.instance_norm(x)
        return x * sigma_projector + mu_style


class DummyResampler(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


def adversarial_loss(d_real, d_fake):
    """

    Args:
        d_real (torch.Tensor): output of disc on real image given real domain
        d_fake (torch.Tensor): output of disc on fake image wihout implicit domain

    Returns:
        torch.Tensor: adversarial loss
    """
    return torch.mean(torch.log(d_real) + torch.log(1 - d_fake))


def style_rec_loss(s, s_rec):
    """

    Args:
        s (torch.Tensor): style code from F (mapping function)
        s_rec (torch.Tensor): style code from style encoder given generated image

    Returns:
        torch.Tensor: loss
    """
    return F.l1_loss(s, s_rec)


def style_div_loss(fake_s1, fake_s2):
    """

    Args:
        fake_s1 (torch.Tensor): fake image given style code 1
        fake_s2 (torch.Tensor): fake image given style code 2

    Returns:
        torch.Tensor: -loss
    """
    return F.l1_loss(fake_s1, fake_s2)


def cycle_loss(true, true_reconstructed):
    """

    Args:
        true (torch.Tensor): source image
        true_reconstructed (torch.Tensor): reversed fake image G(G(x, s), s^hat)

    Returns:
        torch.Tensor: loss
    """
    return F.l1_loss(true, true_reconstructed)
