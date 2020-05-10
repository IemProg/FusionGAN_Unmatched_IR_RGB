import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import time
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt


import Generator
import Discriminator

checkpoint_path = "./model_checkpoints/"

batch_size = 4
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.01
learning_rate_generator = 3e-4
learning_rate_discriminator = 0.1
alpha = 8
beta = 0.001


#####################################################
##				DATASET SETTING					   ##
#####################################################
dataset_dir = './Dataset/'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Need to define dataset

#train_dataset = YouTubePose(datapoint_pairs, shapeLoss_datapoint_pairs, dataset_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=0)


#####################################################
##				MODEL PROTOTYPE					   ##
#####################################################
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
#         if torch.cuda.is_available():
#             x = torch.cuda.FloatTensor(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#Generator & Discriminator
generator = Generator(ResidualBlock)
discriminator = Discriminator(3)

generator = generator.to(device)
discriminator = discriminator.to(device)


#Optimizers
optimizer_gen = optim.SGD(generator.parameters(), lr = learning_rate_generator, momentum=0.9)
optimizer_disc = optim.SGD(discriminator.parameters(), lr = learning_rate_discriminator, momentum=0.9)

def lossIdentity(real_pair, fake_pair):
    batch_size = real_pair.size()[0]
    real_pair = 1 - real_pair
    real_pair = real_pair ** 2
    fake_pair  = fake_pair ** 2
    real_pair = torch.sum(real_pair)
    fake_pair = torch.sum(fake_pair)
    return (real_pair + fake_pair) / batch_size

def lossShape(x, y):
    batch_size = x.size()[0]
    diff = x - y
    diff = diff ** 2
    diff = torch.sum(diff) / batch_size
    return diff


#####################################################
##				MODEL TRAINING					   ##
#####################################################


def save_checkpoint(state, dirpath, epoch):
    filename = 'checkpoint-{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    print('--- checkpoint saved to ' + str(checkpoint_path) + ' ---')

def train_model(gen, disc, loss_i, loss_s, optimizer_gen, optimizer_disc, alpha = 1, beta = 1, num_epochs = 10):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-'*10)
        dataloader = train_dataloader
        gen.train()
        disc.train()
        since = time.time()
        running_loss_iden = 0.0
        running_loss_s1 = 0.0
        running_loss_s2a = 0.0
        running_loss_s2b = 0.0
        running_loss = 0.0
        
        for i_batch, sample_batched in enumerate(dataloader):
            x_gen, y, x_dis = sample_batched['x_gen'], sample_batched['y'], sample_batched['x_dis']
            iden_1, iden_2 = sample_batched['iden_1'], sample_batched['iden_2']
            x_gen = x_gen.to(device)
            y = y.to(device)
            x_dis = x_dis.to(device)
            iden_1 = iden_1.to(device)
            iden_2 = iden_2.to(device)
            
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()
            
            with torch.set_grad_enabled(True):
                x_generated = gen(x_gen, y)
                print('forward 1 done')
                fake_op, fake_pooled_op = disc(x_gen, x_generated)
                real_op, real_pooled_op = disc(x_gen, x_dis)
                loss_identity_gen = -loss_i(real_pooled_op, fake_pooled_op)
                print('Loss calculated')
                loss_identity_gen.backward(retain_graph=True)
                optimizer_gen.step()
                print('backward 1.1 done')
                
                optimizer_disc.zero_grad()
                loss_identity_disc = loss_i(real_op, fake_op)
                print('Loss calculated')
                loss_identity_disc.backward(retain_graph=True)
                optimizer_disc.step()
                print('backward 1.2 done')

                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()
                x_ls2a = gen(y, x_generated)
                x_ls2b = gen(x_generated, y)
                print('forward 2 done')

                loss_s2a = loss_s(y, x_ls2a)
                loss_s2b = loss_s(x_generated, x_ls2b)
                loss_s2 = loss_s2a + loss_s2b
                print('Loss calculated')

                loss_s2.backward()
                optimizer_gen.step()
                print('backward 2 done')

                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()
                
                x_ls1 = generator(iden_1, iden_2)
                print('forward 3 done')

                loss_s1 = loss_s(iden_2, x_ls1)
                print('Loss calculated')
                loss_s1.backward()
                optimizer_gen.step()
                print('backward 5 done')
                print()

            running_loss_iden += loss_identity_disc.item() * x_gen.size(0)
            running_loss_s1 += loss_s1.item() * x_gen.size(0)
            running_loss_s2a += loss_s2a.item() * x_gen.size(0) 
            running_loss_s2b += loss_s2b.item() * x_gen.size(0)
            running_loss = running_loss_iden +  beta * (running_loss_s1 + alpha * (running_loss_s2a + running_loss_s2b))
            print(str(time.time() - since))
            since = time.time()
        epoch_loss_iden = running_loss_iden / dataset_sizes[0]
        epoch_loss_s1 = running_loss_s1 / dataset_sizes[0]
        epoch_loss_s2a = running_loss_s2a / dataset_sizes[0]
        epoch_loss_s2b = running_loss_s2a / dataset_sizes[0]
        epoch_loss = running_loss / dataset_sizes[0]
        print('Identity Loss: {:.4f} Loss Shape1: {:.4f} Loss Shape2a: {:.4f} \
               Loss Shape2b: {:.4f}'.format(epoch_loss_iden, epoch_loss_s1,
                                           epoch_loss_s2a, epoch_loss_s2b))
        print('Epoch Loss: {:.4f}'.format(epoch_loss))

        #Save checkpoints
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'gen_opt': optimizer_gen.state_dict(),
            'disc_opt': optimizer_disc.state_dict()
        }, checkpoint_path, epoch + 1)
        print('Time taken by epoch: {: .0f}m {:0f}s'.format((time.time() - since) // 60, (time.time() - since) % 60))
        print()
        since = time.time()

    return gen, disc

#Run
generator, discriminator = train_model(generator, discriminator, lossIdentity, lossShape, optimizer_gen, optimizer_disc, alpha=alpha, beta=beta, num_epochs = epochs)