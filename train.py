import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from network import HMSNet
from dataset import KITTIDataset

def MSE_loss(pred, gt):
	diff = pred - gt
	mask = (gt > 0).float()
	diff *= mask
	return torch.sum(diff**2) / mask.sum()

def main():
	parser = argparse.ArgumentParser(description='Pytorch HMS-Net example: \nZ. Huang, J. Fan, S. Cheng, S. Yi, X. Wang and H. Li, "HMS-Net: Hierarchical Multi-Scale Sparsity-Invariant Network for Sparse Depth Completion," in IEEE Transactions on Image Processing, vol. 29, pp. 3429-3441, 2020, doi: 10.1109/TIP.2019.2960589.')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=50,
						help='Number of sweeps over the training data')
	parser.add_argument('--frequency', '-f', type=int, default=-1,
						help='Frequency of taking a snapshot')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='./result',
						help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--dataset', '-d', default='data/',
						help='Root directory of dataset')
	parser.add_argument('--cropsize', '-c', type=int, default=None,
						help='Crop size')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	# Set up a neural network to train
	print('set up model')
	net = HMSNet()

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

	# Setup a loss and an optimizer
	print('set up loss, optimizer and scheduler')
	#criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.01)
	scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: (1 - epoch/args.epoch)**0.9)

	print('set up dataset')
	trainset = KITTIDataset(args.dataset, mode="train", cropsize=args.cropsize)
	valset = KITTIDataset(args.dataset, mode="val", cropsize=args.cropsize)
	trainsize = len(trainset)
	valsize = len(valset)
	print(f"trainset: {trainsize}, valset: {valsize}")

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

	# Setup result holder
	x = []
	ac_train = []
	ac_val = []
	# Train
	print('train')
	for ep in range(args.epoch):  # Loop over the dataset multiple times
		net.train()

		running_loss = 0.0
		loss_train = 0
		total_train = 0
		loss_val = 0
		total_val = 0
		num_batch = 0

		for batch in trainloader:
			depths, gts = batch
			masks = (depths > 0).float()

			assert depths.size() == gts.size()

			if args.gpu >= 0:
				depths = depths.to(device=device)
				masks = masks.to(device=device)
				gts = gts.to(device=device)

			optimizer.zero_grad()
			output = net(depths, masks)
			
			loss = MSE_loss(output, gts)
			loss.backward()
			optimizer.step()

			#print(loss.shape, output.shape)

			for i in range(len(output)):
				loss_train += loss.item()
				total_train += 1
			running_loss += loss.item()
			
			#print(f"mini-batch: {num_batch + 1}")
			num_batch += 1

		# Report loss of the epoch
		print('[epoch %d] loss: %.3f\tlr: %.8f' % (ep + 1, running_loss, scheduler.get_last_lr()[0]))

		# Save the model
		if (ep + 1) % args.frequency == 0:
			path = args.out + "/model_" + str(ep + 1)
			torch.save(net.state_dict(), path)

		# Validation
		with torch.no_grad():
			for batch in valloader:
				depths, gts = batch
				masks = (depths > 0).float()
			
				if args.gpu >= 0:
					depths = depths.to(device=device)
					masks = masks.to(device=device)
					gts = gts.to(device=device)

				output = net(depths, masks)
				
				loss = MSE_loss(output,gts)

				for i in range(len(output)):
					loss_val += loss.item()
					total_val += 1
		
		scheduler.step()

		# Record result
		x.append(ep + 1)
		ac_train.append(100 * loss_train / total_train)
		ac_val.append(100 * loss_val / total_val)

	print('Finished Training')
	path = args.out + "/model_final"
	torch.save(net.state_dict(), path)

	# Draw graph
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(x, ac_train, label='Training')
	ax.plot(x, ac_val, label='Validation')
	ax.legend()
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy [%]")
	#ax.set_ylim(0, 100)

	plt.savefig(args.out + '/accuracy.png')
	#plt.show()

if __name__ == '__main__':
	main()
