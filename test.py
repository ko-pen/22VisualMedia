import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from network import HMSNet
from dataset import KITTIDataset

def RMSE(pred, gt):
	diff = pred - gt
	mask = (gt > 0).float()
	diff *= mask
	return (torch.sum(diff**2) / mask.sum()) ** 0.5

def MAE(pred, gt):
	diff = torch.abs(pred - gt)
	mask = (gt > 0).float()
	diff *= mask
	return torch.sum(diff) / mask.sum()

def main():
	parser = argparse.ArgumentParser(description='Pytorch HMS-Net example: \nZ. Huang, J. Fan, S. Cheng, S. Yi, X. Wang and H. Li, "HMS-Net: Hierarchical Multi-Scale Sparsity-Invariant Network for Sparse Depth Completion," in IEEE Transactions on Image Processing, vol. 29, pp. 3429-3441, 2020, doi: 10.1109/TIP.2019.2960589.')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='./result',
						help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--dataset', '-d', default='data/',
						help='Root directory of dataset')
	parser.add_argument('--pass_test', '-p', default=False)
	parser.add_argument('--index', '-i', type=int, default=0)
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('')

	# Set up a neural network to test
	print('set up model')
	net = HMSNet()
	net.load_state_dict(torch.load(args.model))
	
	# Load designated network weight
	if args.resume:
		net.load_state_dict(torch.load(args.resume))
	
	# Set model to GPU
	if args.gpu >= 0:
		print('set model to GPU')
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)
	#else:
	#	print('device: mps')
	#	device = torch.device("mps")
	#	net = net.to(device)

	print('set up dataset')
	testset = KITTIDataset(args.dataset, mode="test")
	testsize = len(testset)
	print(f"testset: {testsize}")
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)

	rmse_sum = 0.0
	mae_sum = 0.0
	total_test = 0
	num_batch = 0

	# Test
	print('test')
	if not args.pass_test:
		net.eval()
		with torch.no_grad():
			for batch in testloader:
				images, depths, gts = batch
				masks = (depths > 0).float()

				assert depths.size() == gts.size()

				if args.gpu >= 0:
					depths = depths.to(device=device)
					masks = masks.to(device=device)
					gts = gts.to(device=device)

				output = net(depths, masks)
				
				rmse = RMSE(output, gts)
				mae = MAE(output, gts)

				rmse_sum += rmse.item()
				mae_sum += mae.item()
				total_test += 1
				
				#print(f"mini-batch: {num_batch + 1}")
				num_batch += 1
			
		print(f"RMSE: {rmse_sum / total_test}, MAE: {mae_sum / total_test}")

	# Draw Output
	image, depth, gt = testset[args.index]
	depth = depth.unsqueeze(dim=0)
	gt = gt.unsqueeze(dim=0)
	mask = (depth > 0).float()

	if args.gpu >= 0:
		depth = depth.to(device=device)
		mask = mask.to(device=device)
		gt = gt.to(device=device)
	
	output = net(depth, mask)
	print(output.size(), torch.min(output), torch.max(output))
	output = output.to('cpu').detach().numpy().copy()[0][0]
	depth = depth.to('cpu').detach().numpy().copy()[0][0]
	gt = gt.to('cpu').detach().numpy().copy()[0][0]

	plt.imsave(args.out + '/image.png', image.astype(np.uint8))
	plt.imsave(args.out + '/depth.png', depth)
	plt.imsave(args.out + '/output.png', output)
	plt.imsave(args.out + '/gt.png', gt)

	fig = plt.figure(figsize=(12,14))
	cmap = plt.cm.gist_ncar
	cmap.set_under(color="black")
	ax1 = fig.add_subplot(4, 1, 1, title="image")
	ax2 = fig.add_subplot(4, 1, 2, title="depth")
	ax3 = fig.add_subplot(4, 1, 3, title="output")
	ax4 = fig.add_subplot(4, 1, 4, title="ground truth")
	im1 = ax1.imshow(image)
	im2 = ax2.imshow(depth, cmap=cmap, vmin=1, vmax=128)
	#fig.colorbar(im2, ax=ax2)
	im3 = ax3.imshow(output, cmap=cmap, vmin=1, vmax=128)
	#fig.colorbar(im3, ax=ax3)
	im4 = ax4.imshow(gt, cmap=cmap, vmin=1, vmax=128)
	#fig.colorbar(im4, ax=ax4)
	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')
	plt.savefig(args.out + '/result.png')
	plt.show()

if __name__ == '__main__':
	main()