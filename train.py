import os
import random


import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import data_utils
from GAN-MLP import Discriminator, Generator, GAN

#BATCH_SIZE = 64
FLATTENED_DIMENSION = 784
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def MnistDataLoaders(data_path, batch_size):
    transform = lambda x : (x - 127.5)/(127.5)
    dataset = data_utils.MnistDataset(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def training_gan(data_path, epochs = 100, batch_size = 64, k = 1):
	### Need to initializa the networks
	### Set the optimizers
	### Write the training loops
	discrimator = Discriminator().to(DEVICE)
	generator = Generator().to(DEVICE)
	gan = GAN(generator, discrimator).to(DEVICE)

	optim_discriminator = torch.optim.SGD(discrimator.parameters(), lr = 0.001, momentum = 0.9)
	optim_gan = torch.optim.SGD(gan.parameters(), lr = 0.001, momentum = 0.9)

	loss_func = torch.nn.CrossEntropyLoss()

	data_loader = MnistDataLoaders(data_path)
	for epoch in range(epochs):
		### update the disciminator weights here
		loss_discriminator = []
		loss_gans = []
		for it in range(200):
			for i in range(k):
				random_noise = torch.from_numpy(np.random.randn(batch_size, FLATTENED_DIMENSION)).to(DEVICE)
				fake_images = generator(random_noise)

				for data in data_loader:
					actual_images, _ = data
					break
				
				actual_image = actual_images.to(DEVICE)
				inputs = torch.cat((fake_images, actual_image))

				labels = torch.zeros(2*batch_size)
				labels[batch_size:] = 1
				labels = labels.to(DEVICE)
			
				disc_logits = discrimator(inputs)


				loss_disc = loss_func(logits, labels)
				loss_disc.backward()
				loss_discriminator.append(loss.item())

				optim_discriminator.step()


			print('Epoch {}: Iteration {}/{} - Discriminator Loss {}'.format(epoch, it, 200, np.mean(loss_discriminator)))
			# update the gan here

			## freeze the discriminator here
			for param in discrimator.parameters():
				param.requires_grad = False
			
			random_noise = torch.from_numpy(np.random.randn(batch_size, FLATTENED_DIMENSION)).to(DEVICE)
			labels_gan = torch.ones(batch_size).to(DEVICE)

			gan_logits = gan(random_noise)

			loss_gan = loss_func(gan_logits, labels_gan)
			loss_gans.append(loss_gan.item())
			loss_gen.backward()

			optim_gan.step()

			print('Epoch {}: Iteration {}/{} - GAN Loss {}'.format(epoch, it, 200, np.mean(loss_gans)))

			## unfreeze the discriminator
			for param in discrimator.parameters():
				param.requires_grad = True

		
	return generator
	

		





