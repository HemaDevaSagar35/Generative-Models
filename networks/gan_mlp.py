import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Network architecture for discriminator.
    At this stage we are building with MLP blocks
    Architecture design: 
    -----> input (784) - 
    -----> hidden 1 (1024) - 
    -----> hidden 2 (512) - 
    -----> hidden 3 (256) - 
    -----> output (2)

    Notes:
    1) I am not involving dropouts as of now, cause I want to see how overfitting happens for the discriminator here.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = nn.Linear(784, 1024)
        self.batch_norm_1 = nn.BatchNorm1d(1024)

        self.layer_2 = nn.Linear(1024, 512)
        self.batch_norm_2 = nn.BatchNorm1d(512)

        self.layer_3 = nn.Linear(512, 256)
        self.batch_norm_3 = nn.BatchNorm1d(256)

        self.layer_4 = nn.Linear(256, 2)
    
    def forward(self, input):
        output = nn.LeakyReLU()(self.layer_1(input))
        output = self.batch_norm_1(output)
        print(output.std())

        output = nn.LeakyReLU()(self.layer_2(output))
        output = self.batch_norm_2(output)
        print(output.std())

        output = nn.LeakyReLU()(self.layer_3(output))
        output = self.batch_norm_3(output)
        print(output.std())
        
        output = self.layer_4(output)
        print(output)

        return output



class Generator(nn.Module):
    """Network architecture for generator. At this stage
    we are building with MLP blocks
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.layer_1 = nn.Linear(100, 256)
        
        self.layer_2 = nn.Linear(256, 512)
        self.layer_3 = nn.Linear(512,1024)
        self.layer_4 = nn.Linear(1024, 784)
    
    def forward(self, input):
        output = nn.LeakyReLU()(self.layer_1(input))
        output = nn.LeakyReLU()(self.layer_2(output))
        output = nn.LeakyReLU()(self.layer_3(output))
        output = nn.Tanh()(self.layer_4(output))
        
        return output


class GAN(nn.Module):
    """This combines generator with discriminator and discriminator
    is tagged false for trainable"""
    #Note: Be careful with the freezing of the discriminator layer
    def __init__(self, generator, discrimator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discrimator
        for param in self.discriminator.parameters():
            param.requires_grad = False
  
    def forward(self, input):
        output = self.generator(input)
        output = self.discriminator(output)
        return output




