#!/usr/bin/env python3

from monai.utils import set_determinism
from monai.networks.nets import GlobalNet, LocalNet, RegUNet, unet
from monai.config import USE_COMPILED
from monai.networks.blocks import Warp, DVF2DDF
import torch
from torch.nn import MSELoss
from monai.transforms import (
    LoadImage,
    Resize,
    EnsureChannelFirst,
    ScaleIntensityRangePercentiles,
)
#from monai.data.nifti_writer import write_nifti
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import (
    GlobalMutualInformationLoss,
    LocalNormalizedCrossCorrelationLoss,
)
from nilearn.image import resample_to_img, resample_img, crop_img, load_img
from torch.nn.functional import grid_sample
from warp_utils import get_grid, apply_warp, jacobian_determinant
from typing import List
from monai.losses import BendingEnergyLoss
from deform_losses import BendingEnergyLoss as myBendingEnergyLoss
from deform_losses import GradEnergyLoss
from networks import LocalNet2
import argparse
import nibabel as nib


class dscolors:
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    purple = "\033[95m"
    cyan = "\033[96m"
    clear = "\033[0m"
    bold = "\033[1m"
    ul = "\033[4m"


class Warper:
    # device = 'cuda'
    # max_epochs = 3000
    # lr = .01
    def __init__(self):
        set_determinism(42)

    # def setLoss(self, loss):
    # 	self.loss=loss
    # 	if loss == 'mse':
    # 		image_loss = MSELoss()
    # 	elif loss == 'cc':
    # 		image_loss = LocalNormalizedCrossCorrelationLoss()
    # 	elif loss == 'mi':
    # 		image_loss = GlobalMutualInformationLoss()
    # 	else:
    # 		AssertionError

    # 	set_determinism(42)

    def loadMoving(self, moving_file):
        self.moving, self.moving_meta = LoadImage()(moving_file)
        self.moving = EnsureChannelFirst()(self.moving)

    def loadTarget(self, fixed_file):
        self.target, self.target_meta = LoadImage()(fixed_file)
        self.target = EnsureChannelFirst()(self.target)

    def loadTargetMask(self, target_mask):
        if target_mask is None:
            self.target_mask = None
        else:
            self.target_mask, self.target_mask_meta = LoadImage()(target_mask)
            self.target_mask = self.target_mask > 0.5
            self.target_mask.type(torch.DoubleTensor)
            self.target_mask = EnsureChannelFirst()(self.target_mask)

    def saveWarpedLabels(self, label_file, output_label_file):
        print(dscolors.green + "warping " + label_file + dscolors.clear)
        print(
            dscolors.green
            + "saving warped labels: "
            + dscolors.clear
            + output_label_file
            + dscolors.clear
        )
        label, meta = LoadImage()(label_file)
        label = EnsureChannelFirst()(label)
        warped_labels = apply_warp(
            self.ddf[None,], label[None,], self.target[None,], interp_mode="nearest"
        )
        #write_nifti(warped_labels[0, 0], output_label_file, affine=self.target.affine)
        nib.save(nib.Nifti1Image(warped_labels[0, 0].detach().cpu().numpy(), self.target.affine), output_label_file)

    def nonlinear_reg(
        self,
        target_file,
        moving_file,
        output_file,
        target_mask=None,
        label_file=None,
        ddf_file=None,
        output_label_file=None,
        jacobian_determinant_file=None,
        loss="cc",
        reg_penalty=0.3,
        nn_input_size=64,
        lr=1e-6,
        max_epochs=1000,
        device="cuda",
    ):
        if loss == "mse":
            image_loss = MSELoss()
        elif loss == "cc":
            image_loss = LocalNormalizedCrossCorrelationLoss()
        elif loss == "mi":
            image_loss = GlobalMutualInformationLoss()
        else:
            raise AssertionError("Invalid Loss")

        regularization = myBendingEnergyLoss()  # GradEnergyLoss()
        #######################
        set_determinism(42)
        self.loadMoving(moving_file)
        self.loadTarget(target_file)
        self.loadTargetMask(target_mask)

        SZ = nn_input_size
        moving_ds = Resize(spatial_size=[SZ, SZ, SZ], mode="trilinear")(self.moving).to(
            device
        )
        target_ds = Resize(spatial_size=[SZ, SZ, SZ], mode="trilinear")(self.target).to(
            device
        )

        if target_mask is not None:
            target_mask_ds = Resize(spatial_size=[SZ, SZ, SZ], mode="trilinear")(self.target_mask).to(device)


        moving_ds = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True
        )(moving_ds)
        target_ds = ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True
        )(target_ds)
        reg = unet.UNet(
            spatial_dims=3,  # spatial dims
            in_channels=2,
            out_channels=3,  # output channels (to represent 3D displacement vector field)
            channels=(16, 32, 32, 32, 32),  # channel sequence
            strides=(1, 2, 2, 4),  # convolutional strides
            dropout=0.2,
            norm="batch",
        ).to(device)
        if USE_COMPILED:
            warp_layer = Warp(3, padding_mode="zeros").to(device)
        else:
            warp_layer = Warp("bilinear", padding_mode="zeros").to(device)
        reg.train()
        optimizerR = torch.optim.Adam(reg.parameters(), lr=lr)
        print(dscolors.green + "optimizing" + dscolors.clear)

        dvf_to_ddf = DVF2DDF()

        if target_mask is not None:
            target_ds *= target_mask_ds
        
        for epoch in range(max_epochs):
            optimizerR.zero_grad()
            input_data = torch.cat((moving_ds, target_ds), dim=0)
            input_data = input_data[None,]
            dvf_ds = reg(input_data)
            ddf_ds = dvf_to_ddf(dvf_ds)
            image_moved = warp_layer(moving_ds[None,], ddf_ds)
            
            if target_mask is not None:
                image_moved *= target_mask_ds

            imgloss = image_loss(image_moved, target_ds[None,])
            regloss = reg_penalty * regularization(ddf_ds)
            vol_loss = imgloss + regloss

            # print('imgloss:'+dscolors.blue+f'{imgloss:.4f}'+dscolors.clear
            # 			+', regloss:'+dscolors.blue+f'{regloss:.4f}'+dscolors.clear)#, end=' ')
            vol_loss.backward()
            optimizerR.step()
            # print('epoch_loss:'+dscolors.blue+f'{vol_loss:.4f}'+dscolors.clear
            # 		+' for epoch:'+dscolors.blue+f'{epoch}'+'/'+f'{max_epochs}'+dscolors.clear+'     ',end='\r\033[A')
            print(
                "epoch:",
                dscolors.green,
                f"{epoch}/{max_epochs}",
                "Loss:",
                dscolors.yellow,
                f"{vol_loss.detach().cpu().numpy():.2f}",
                dscolors.clear,
                "",
                end="\r",
            )

        print("finished", dscolors.green, f"{max_epochs}", dscolors.clear, "epochs")

        print("\n\n")
        # write_nifti(image_moved[0, 0], 'moved_ds.nii.gz', affine=target_ds.affine)
        # write_nifti(target_ds[0], 'target_ds.nii.gz', affine=target_ds.affine)
        # write_nifti(moving_ds[0], 'moving_ds.nii.gz', affine=target_ds.affine)
        # write_nifti(torch.permute(ddf_ds[0],[1,2,3,0]),'ddf_ds.nii.gz',affine=target_ds.affine)
        # jdet_ds = jacobian_determinant(ddf_ds[0])
        # write_nifti(jdet_ds,'jdet_ds.nii.gz',affine=target_ds.affine)

        print(dscolors.green + "computing deformation field" + dscolors.clear)
        size_moving = self.moving[0].shape
        size_target = self.target[0].shape
        ddfx = Resize(spatial_size=size_target, mode="trilinear")(ddf_ds[:, 0]) * (
            size_moving[0] / SZ
        )
        ddfy = Resize(spatial_size=size_target, mode="trilinear")(ddf_ds[:, 1]) * (
            size_moving[1] / SZ
        )
        ddfz = Resize(spatial_size=size_target, mode="trilinear")(ddf_ds[:, 2]) * (
            size_moving[2] / SZ
        )
        self.ddf = torch.cat((ddfx, ddfy, ddfz), dim=0)
        del ddf_ds, ddfx, ddfy, ddfz
        # Apply the warp
        print(dscolors.green + "applying warp" + dscolors.clear)
        image_movedo = apply_warp(
            self.ddf[None,], self.moving[None,], self.target[None,]
        )
        print(dscolors.green + "saving warped output: " + dscolors.clear + output_file)
        # write_nifti(image_movedo[0, 0], output_file, affine=self.target.affine)
        nib.save(
            nib.Nifti1Image(
                image_movedo[0, 0].detach().cpu().numpy(), self.target.affine
            ),
            output_file,
        )

        if ddf_file is not None:
            print(dscolors.green + "saving ddf: " + dscolors.clear + ddf_file)
            nib.save(
                nib.Nifti1Image(
                    torch.permute(self.ddf, [1, 2, 3, 0]).detach().cpu().numpy(),
                    self.target.affine,
                ),
                ddf_file,
            )

        # Apply the warp to labels
        if label_file is not None and output_label_file is not None:
            print(dscolors.green + "warping " + label_file + dscolors.clear)
            print(
                dscolors.green
                + "saving warped labels: "
                + dscolors.clear
                + output_label_file
                + dscolors.clear
            )
            label, meta = LoadImage()(label_file)
            label = EnsureChannelFirst()(label)
            warped_labels = apply_warp(
                self.ddf[None,], label[None,], self.target[None,], interp_mode="nearest"
            )
            # write_nifti(warped_labels[0,0], output_label_file, affine=self.target.affine)
            nib.save(
                nib.Nifti1Image(
                    warped_labels[0, 0].detach().cpu().numpy(), self.target.affine
                ),
                output_label_file,
            )

        if jacobian_determinant_file is not None:
            jdet = jacobian_determinant(self.ddf)
            # write_nifti(jdet,'jdet.nii.gz',affine=self.target.affine)
            nib.save(
                nib.Nifti1Image(jdet, self.target.affine), jacobian_determinant_file
            )


#####################
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Nonlinear registration for mouse brains"
    )
    parser.add_argument("moving_file", type=str, help="moving file name")
    parser.add_argument("target_file", type=str, help="fixed file name")
    parser.add_argument("output_file", type=str, help="output file name")
    parser.add_argument(
        "--label-file", "--label", type=str, help="input label file name"
    )
    parser.add_argument("--output-label-file", type=str, help="output label file name")
    parser.add_argument("-j", "--jacobian", type=str, help="output jacobian file name")
    parser.add_argument(
        "-ddf",
        "--ddf-file",
        type=str,
        default="",
        help="dense displacement field file name",
    )
    parser.add_argument(
        "--nn_input_size",
        type=int,
        default=64,
        help="size of the neural network input (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "-e", "--max-epochs", type=int, default=3000, help="maximum interations"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="device: cuda, cpu, etc."
    )
    parser.add_argument(
        "-l", "--loss", type=str, default="cc", help="loss function: mse, cc or mi"
    )
    parser.add_argument(
        "-r",
        "--reg-penalty",
        type=str,
        default=0.3,
        help="loss function: mse, cc or mi",
    )
    # parser.add_argument('--')

    args = parser.parse_args()
    warper = Warper()

    warper.nonlinear_reg(
        target_file=args.fixed_file,
        moving_file=args.moving_file,
        output_file=args.output_file,
        ddf_file=args.ddf_file,
        label_file=args.label_file,
        output_label_file=args.output_label_file,
        jacobian_determinant_file=args.jacobian,
        loss=args.loss,
        reg_penalty=args.reg_penalty,
        nn_input_size=args.nn_input_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
