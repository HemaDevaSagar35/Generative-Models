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
        self.layer_2 = nn.Linear(1024, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(2)
    
    def forward(self, input):
        output = nn.LeakyReLU(self.layer_1(input))
        output = nn.LeakyReLU(self.layer_2(output))
        output = nn.LeakyReLU(self.layer_3(output))
        output = self.layer_4(output)

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
        output = nn.LeakyReLU(self.layer_1(input))
        output = nn.LeakyReLU(self.layer_2(output))
        output = nn.LeakyReLU(self.layer_3(output))
        output = nn.tanh(self.layer_4(output))
        
        return output


class GAN(nn.Module):
    """This combines generator with discriminator and discriminator
    is tagged false for trainable"""



