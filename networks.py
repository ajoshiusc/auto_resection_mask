from monai.networks.nets.regunet import RegUNet
from monai.networks.blocks.regunet_block import (
    RegistrationDownSampleBlock,
    RegistrationExtractionBlock,
    RegistrationResidualConvBlock,
    get_conv_block,
    get_deconv_block,
)
from typing import List, Optional, Tuple, Union
from torch import nn
from torch.nn import functional as F
import torch


class LocalNet2(RegUNet):
    """
    Reimplementation of LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        extract_levels: Tuple[int],
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        pooling: bool = True,
        depth=3,
        use_addictive_sampling: bool = True,
        concat_skip: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            use_addictive_sampling: whether use additive up-sampling layer for decoding.
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
            depth: depth of the network
        """
        self.use_additive_upsampling = use_addictive_sampling
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            extract_levels=extract_levels,
            depth=depth,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=[7] + [3] * max(extract_levels),
        )

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return get_conv_block(
            spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.use_additive_upsampling:
            return AdditiveUpSampleBlock(
                spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels
            )

        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)
