import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import data_utils
from networks import gan_mlp #Discriminator, Generator, GAN

#BATCH_SIZE = 64
LATENT_DIMENSION = 100
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
	discrimator = gan_mlp.Discriminator().to(DEVICE)
	generator = gan_mlp.Generator().to(DEVICE)
	gan = gan_mlp.GAN(generator, discrimator).to(DEVICE)
	
	optim_gan = torch.optim.SGD(gan.generator.parameters(), lr = 0.001, momentum = 0.9)
	for param in discrimator.parameters():
		param.requires_grad = True

	optim_discriminator = torch.optim.SGD(discrimator.parameters(), lr = 0.001, momentum = 0.9)
	#optim_gan = torch.optim.SGD(gan.parameters(), lr = 0.001, momentum = 0.9)

	loss_func = torch.nn.CrossEntropyLoss()

	data_loader = MnistDataLoaders(data_path, batch_size)
	for epoch in range(epochs):
		### update the disciminator weights here
		# loss_discriminator = []
		# loss_gans = []
		for it in range(200):
			loss_discriminator = []
			loss_gans = []
			for i in range(k):
				random_noise = torch.from_numpy(np.random.randn(batch_size, LATENT_DIMENSION).astype(np.float32)).to(DEVICE)
				#print(random_noise.dtype)
				
				fake_images = generator(random_noise)
				#print(fake_images.shape)
				for data in data_loader:
					actual_images, _ = data
					break
				
				actual_image = actual_images.float().to(DEVICE)
				#print(actual_image.dtype)
				inputs = torch.cat((fake_images, actual_image))

				labels = torch.zeros(2*batch_size, dtype = torch.long)
				labels[batch_size:] = 1
				labels = labels.to(DEVICE)
			
				disc_logits = discrimator(inputs)

				#print(labels.dtype)
				loss_disc = loss_func(disc_logits, labels)
				loss_disc.backward()
				loss_discriminator.append(loss_disc.item())

				optim_discriminator.step()

			#### checking if discriminator weights are changing
			print("DDDDDDDDDDDDDDDDDDDDDD")
			for param in discrimator.parameters():
				print(param.data.std())
			print("DDDDDDDDDDDDDDDDDDDDDDD")
			print('Epoch {}: Iteration {}/{} - Discriminator Loss {}'.format(epoch+1, it+1, 200, np.mean(loss_discriminator)))
			# update the gan here

			## freeze the discriminator here
			for param in discrimator.parameters():
				param.requires_grad = False
			
			random_noise = torch.from_numpy(np.random.randn(batch_size, LATENT_DIMENSION).astype(np.float32)).to(DEVICE)
			labels_gan = torch.ones(batch_size, dtype = torch.long).to(DEVICE)

			gan_logits = gan(random_noise)

			loss_gan = loss_func(gan_logits, labels_gan)
			loss_gans.append(loss_gan.item())
			loss_gan.backward()

			optim_gan.step()

			print('Epoch {}: Iteration {}/{} - GAN Loss {}'.format(epoch+1, it+1, 200, np.mean(loss_gans)))

			# print("GGGGGGGGGGGGGGGGGGGGGG")
			# for param in discrimator.parameters():
			# 	print(param.data.mean())
			# print("GGGGGGGGGGGGGGGGGGGGGG")

			# print("AAAAAAAAAAAAAAAAAAAAAA")
			# for param in gan.discriminator.parameters():
			# 	print(param.data.mean())
			# print("AAAAAAAAAAAAAAAAAAAAAAA")

			# print("GANGANGANGANGANGANGANGAN")
			# for param in gan.generator.parameters():
			# 	print(param.data.mean())
			# print("GANGANGANGANGANGANGANGAN")

			## unfreeze the discriminator
			for param in discrimator.parameters():
				param.requires_grad = True

			
		
	return generator
	

		





