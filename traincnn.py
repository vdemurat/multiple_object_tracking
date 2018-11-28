import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from datautils import PreTrainedResnet, Pixel

from dataloader import get_train_data

import torchvision
import torchvision.models as models


class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()

		self.pretrained_model = models.resnet18(pretrained=True)
		if torch.cuda.is_available():
			self.pretrained_model = self.pretrained_model.cuda()
		for param in self.pretrained_model.parameters():
			param.requires_grad = True

		self.linear = nn.Linear(1000, 500)

	def forward(self, x):
		for name, module in self.pretrained_model._modules.items():
			if name=='fc':
				x = x.squeeze().contiguous()
			x = module(x)
			if name=='fc':
				lastlayer = x

		features = self.linear(lastlayer)
		return features


class SiemeseCNN(nn.Module):

	def __init__(self):
		super(SiemeseCNN, self).__init__()
		self.cnn = CNN()
		self.linear1 = nn.Linear(1000, 500)
		self.linear2 = nn.Linear(500, 2)

	def forward(self, x1, x2):
		feat1 = self.cnn(x1)
		feat2 = self.cnn(x2)
		return self.linear2(functional.ReLU(self.linear1(torch.cat((feat1, feat2), dim=-1))))


def train_cnn():

	modeltype = 'cnn'
	train_batch_size = 2
	sequence_length = 1
	directory_type = 'DPM'

	pixel = Pixel('pixel.pkl')

	train_dataloader, train_data_size, val_dataloader, val_data_size = \
		get_train_data(train_batch_size, sequence_length, directory_type, pixel)

	cnnmodel = CNN()
	if torch.cuda.is_available(): cnnmodel = cnnmodel.cuda()

	for i, batch in enumerate(train_dataloader):
		frame1 = Variable(torch.stack(batch[0])).squeeze(1)
		frame2 = Variable(torch.stack(batch[1]))
		labels = Variable(torch.LongTensor(batch[2]))

		print(frame1.size())
		print(frame2.size())
		print(labels)

		print(cnnmodel(frame1).size())
		break


if __name__=='__main__':
	train_cnn()