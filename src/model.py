import torch.nn as nn
import torch
import torch.nn.functional as F


class DummyResampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        self.out_conv = nn.Conv2d(64, 3, 1, padding=0)
        self.downsampling = nn.Sequential(
            # 64x64
            ResBlock(64, 128, nn.InstanceNorm2d(64)),
            nn.AvgPool2d(2),
            # 32x32
            ResBlock(128, 256, nn.InstanceNorm2d(128)),
            nn.AvgPool2d(2),
            # 16x16
            ResBlock(256, 512, nn.InstanceNorm2d(256)),
            nn.AvgPool2d(2),
            # 8x8
            ResBlock(512, 512, nn.InstanceNorm2d(512)),
            nn.AvgPool2d(2),
            # 4x4
            ResBlock(512, 512, nn.InstanceNorm2d(512)),
        )
        self.intermediate1 = nn.Sequential(
            ResBlock(512, 512, nn.InstanceNorm2d(512)),
            ResBlock(512, 512, nn.InstanceNorm2d(512)),
        )

        self.intermediate2 = nn.ModuleList(
            [
                AdaINResBlock(512, 512),
                AdaINResBlock(512, 512),
            ]
        )

        self.upsampling = nn.ModuleList(
            [
                # 4x4
                nn.Upsample(scale_factor=2),
                AdaINResBlock(512, 256),
                # 8x8
                nn.Upsample(scale_factor=2),
                AdaINResBlock(256, 128),
                # 16x16
                nn.Upsample(scale_factor=2),
                AdaINResBlock(128, 64),
                # 32x32
                nn.Upsample(scale_factor=2),
                AdaINResBlock(64, 64),
                # 64x64
            ]
        )

    def forward(self, x, s):
        x = self.in_conv(x)
        x = self.downsampling(x)
        x = self.intermediate1(x)
        for layer in self.intermediate2:
            x = layer(x, s)
        for layer in self.upsampling:
            x = layer(x, s) if isinstance(layer, AdaINResBlock) else layer(x)
        return self.out_conv(x)


class Discriminator(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.downsampling = nn.Sequential(
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
            nn.Conv2d(512, 512, 2, padding=0),
            nn.LeakyReLU(),
        )
        self.out = nn.ModuleList([nn.Linear(512, 1) for _ in range(K)])

    def forward(self, x, y):
        x = self.in_conv(x)
        x = self.downsampling(x)
        x = self.conv(x)
        x = x.squeeze(3).squeeze(2)
        output = self.out[y](x)
        return output


class MappingNetwork(nn.Module):
    """Generates style codes \mathcal{s} for every domain \mathcal{y}
    given some latent code \mathcal{z}. Used during training for generating
    images with random a) domain and b) style. Trained with generator via
    adversarial objective and style diversificaton loss
    """

    def __init__(self, K, D):
        super().__init__()
        self.backbone = nn.Sequential(
            self.get_fc_block(16, 512),
            self.get_fc_block(512, 512),
            self.get_fc_block(512, 512),
            self.get_fc_block(512, 512),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    self.get_fc_block(512, 512),
                    self.get_fc_block(512, 512),
                    self.get_fc_block(512, 512),
                    self.get_fc_block(512, D, activation=False),
                )
                for j in range(K)
            ]
        )

    def forward(self, z, y):
        z = self.backbone(z)
        # select y's head corresponding to
        output = self.heads[y](z)
        return output

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

    def __init__(self, D, K):
        super().__init__()
        self.in_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.downsampling = nn.Sequential(
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
            nn.Conv2d(512, 512, 2, padding=0),
            nn.LeakyReLU(),
        )
        self.out = nn.ModuleList([nn.Linear(512, D) for _ in range(K)])

    def forward(self, x, y):
        x = self.in_conv(x)
        x = self.downsampling(x)
        x = self.conv(x)
        x = x.squeeze(3).squeeze(2)
        output = self.out[y](x)
        return output


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
        act=nn.ReLU(),
    ):
        super().__init__()
        self.norm = norm
        self.act = act
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.out_channels != self.in_channels:
            self.proj_conv = nn.Conv2d(
                in_channels, out_channels, 1, padding=0, bias=False
            )

        self.layers = nn.Sequential(
            self.norm,
            self.act,
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            self.norm,
            self.act,
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x, s=None):
        return self.shortcut(x) + self.layers(x)

    def shortcut(self, x):
        if self.out_channels != self.in_channels:
            return self.proj_conv(x)
        return x


class AdaINResBlock(nn.Module):
    """
    Building block for Generator

    Args:
        norm (nn.Module): Instance norm or Adaptive Instance norm
        act (nn.Module): Activation function
        pre_resampler (nn.Module): ConvTranspoce or Upsample for upsampling blocks
    """

    def __init__(self, in_channels, out_channels, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.style_dim = 64
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.out_channels != self.in_channels:
            self.proj_conv = nn.Conv2d(
                in_channels, out_channels, 1, padding=0, bias=False
            )

        self.layers = nn.ModuleList(
            [
                AdaIN(in_channels, self.style_dim),
                self.act,
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                AdaIN(in_channels, self.style_dim),
                self.act,
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            ]
        )

    def forward(self, x, s):
        resid = x
        for layer in self.layers:
            x = layer(x) if not isinstance(layer, AdaIN) else layer(x, s)
        return self.shortcut(resid) + x

    def shortcut(self, x):
        if self.out_channels != self.in_channels:
            return self.proj_conv(x)
        return x


class AdaIN(nn.Module):
    """Like in style gan"""

    def __init__(self, n_fmaps, style_dim):
        super().__init__()
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


def adversarial_loss(d_out: torch.Tensor, label: int):
    """

    Args:
        d_out (torch.Tensor): output of disc on real image given real domain
        label (int): output of disc on fake image wihout implicit domain

    Returns:
        torch.Tensor: adversarial loss
    """
    labels = torch.ones_like(d_out) if label == 1 else torch.zeros_like(d_out)
    return F.binary_cross_entropy_with_logits(d_out, labels)


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
