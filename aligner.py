#!/usr/bin/env python3

from monai.utils import set_determinism
from monai.networks.nets import GlobalNet
from monai.config import USE_COMPILED
from monai.networks.blocks import Warp
from torch.nn import MSELoss
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, ScaleIntensityRangePercentiles
from monai.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from warp_utils import apply_warp
import argparse
import torch
import nibabel as nib








class dscolors:
	red  	 = '\033[91m'
	green  = '\033[92m'
	yellow = '\033[93m'
	blue   = '\033[94m'
	purple = '\033[95m'
	cyan   = '\033[96m'
	clear  = '\033[0m'
	bold   = '\033[1m'
	ul     = '\033[4m'

class Aligner:
	image_loss = MSELoss()
	nn_input_size=64
	lr=1e-6
	max_epochs=5000
	device='cuda'

	def __init__(self):
		set_determinism(42)

	def setLoss(self, loss):
		self.loss=loss
		if loss == 'mse':
			self.image_loss = MSELoss()
		elif loss == 'cc':
			self.image_loss = LocalNormalizedCrossCorrelationLoss()
		elif loss == 'mi':
			self.image_loss = GlobalMutualInformationLoss()
		else:
			raise AssertionError('Invalid Loss')
			

	def loadMoving(self, moving_file):
		self.moving, self.moving_meta = LoadImage()(moving_file)
		self.moving = EnsureChannelFirst()(self.moving)

	def loadTarget(self, fixed_file):
		self.target, self.moving_meta = LoadImage()(fixed_file)
		self.target = EnsureChannelFirst()(self.target)

	def performAffine(self):
		SZ = self.nn_input_size
		moving_ds = Resize(spatial_size=[SZ, SZ, SZ])(self.moving).to(self.device)
		target_ds = Resize(spatial_size=[SZ, SZ, SZ])(self.target).to(self.device)
		moving_ds = ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(moving_ds)
		target_ds = ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(target_ds)

    # GlobalNet is a NN with Affine head
		reg = GlobalNet(
			image_size=(SZ, SZ, SZ),
			spatial_dims=3,
			in_channels=2,  # moving and fixed
			num_channel_initial=2,
			depth=2).to(self.device)

		if USE_COMPILED:
			warp_layer = Warp(3, padding_mode="zeros").to(self.device)
		else:
			warp_layer = Warp("bilinear", padding_mode="zeros").to(self.device)

		reg.train()
		optimizerR = torch.optim.Adam(reg.parameters(), lr=1e-6)

		for epoch in range(self.max_epochs):
			optimizerR.zero_grad()
			input_data = torch.cat((moving_ds, target_ds), dim=0)
			input_data = input_data[None, ]
			ddf_ds = reg(input_data)
			image_moved = warp_layer(moving_ds[None, ], ddf_ds)
			vol_loss = self.image_loss(image_moved, target_ds[None, ])
			vol_loss.backward()
			optimizerR.step()
			print('epoch_loss:',dscolors.blue,f'{vol_loss.cpu().detach().numpy():.4f}',dscolors.clear,
				' for epoch:', dscolors.green,f'{epoch}/{self.max_epochs}',dscolors.clear, '',end='\r')
            
		size_moving = self.moving[0].shape
		size_target = self.target[0].shape
		ddfx = Resize(spatial_size=size_target, mode='trilinear')(ddf_ds[:, 0])*(size_moving[0]/SZ)
		ddfy = Resize(spatial_size=size_target, mode='trilinear')(ddf_ds[:, 1])*(size_moving[1]/SZ)
		ddfz = Resize(spatial_size=size_target, mode='trilinear')(ddf_ds[:, 2])*(size_moving[2]/SZ)
		self.ddf = torch.cat((ddfx, ddfy, ddfz), dim=0)
		del ddf_ds, ddfx, ddfy, ddfz

	def saveDeformationField(self,ddf_file):
		nib.save(nib.Nifti1Image(torch.permute(self.ddf, [1, 2, 3, 0]).detach().cpu().numpy(), self.target.affine), ddf_file)

	def saveWarpedFile(self,output_file):
    # Apply the warp
		image_movedo = apply_warp(self.ddf[None, ], self.moving[None, ], self.target[None, ])
		nib.save(nib.Nifti1Image(image_movedo[0, 0].detach().cpu().numpy(), self.target.affine), output_file)

	def affine_reg(self, fixed_file, moving_file, output_file, ddf_file, loss='mse', nn_input_size=64, lr=1e-6, max_epochs=5000, device='cuda'):
		self.setLoss(loss)
		self.nn_input_size=nn_input_size
		self.lr=lr,
		self.max_epochs=max_epochs
		self.device=device
		self.loadMoving(moving_file)
		self.loadTarget(fixed_file)
		self.performAffine()
		self.saveWarpedFile(output_file)

		if ddf_file is not None:
			self.saveDeformationField(ddf_file)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Affine registration for mouse brains')

    parser.add_argument('moving_file', type=str, help='moving file name')
    parser.add_argument('fixed_file', type=str, help='fixed file name')
    parser.add_argument('output_file', type=str, help='output file name')
    parser.add_argument('-ddf', '--ddf-file', type=str, help='dense displacement field file name')
    parser.add_argument('--nn_input_size', type=int, default=64, help='size of the neural network input (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-4)')
    parser.add_argument('-e', '--max-epochs', type=int, default=1500, help='maximum interations')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device: cuda, cpu, etc.')
    parser.add_argument('-l', '--loss', type=str, default='mse', help='loss function: mse, cc or mi')

    args = parser.parse_args()
    #print(args)
    aligner=Aligner()
    aligner.affine_reg(fixed_file=args.fixed_file, moving_file=args.moving_file, output_file=args.output_file, ddf_file=args.ddf_file,
               loss=args.loss, nn_input_size=args.nn_input_size, lr=args.lr, max_epochs=args.max_epochs, device=args.device)

if __name__ == "__main__":
    main()
