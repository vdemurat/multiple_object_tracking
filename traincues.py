import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from appearancemodel import AppearanceModel
from datautils import PreTrainedResnet, Pixel

from dataloader import get_train_data

def train_appearance(use_pickle):

	modeltype = 'appearance'
	train_batch_size = 64
	sequence_length = 6
	directory_type = 'DPM'

	pixel = Pixel('pixel.pkl')
	if use_pickle: pixel.load()
	pretrained = PreTrainedResnet({'intermediate_layers':['fc']})
	if torch.cuda.is_available():
		pretrained = pretrained.cuda()

	train_dataloader, train_data_size, val_dataloader, val_data_size = \
		get_train_data(train_batch_size, sequence_length, directory_type, pixel, pretrained)

	save_dir = 'models/{}/'.format(modeltype)
	valfile = open('val.txt', 'a')
	
	dict_args = {
			'input_dim' : 500,
			'rnn_hdim' : 200,
			'rnn_type' : 'LSTM',
			'feature_dim' : 100,
		}

	appmodel = AppearanceModel(dict_args)

	num_epochs = 300
	learning_rate = 1.0
	criterion = nn.NLLLoss() 
	optimizer = optim.Adadelta(appmodel.parameters(), lr=learning_rate, rho=0.95, eps=1e-06, weight_decay=0)
	
	if torch.cuda.is_available():
		appmodel = appmodel.cuda()
		criterion = criterion.cuda()

	for epoch in range(num_epochs):
		for i, batch in enumerate(train_dataloader):
			trackframes = Variable(torch.stack(batch[0]))
			detectionframe = Variable(torch.stack(batch[1]))
			labels = Variable(torch.LongTensor(batch[2]))

			if torch.cuda.is_available():
				trackframes = trackframes.cuda()
				detectionframe = detectionframe.cuda()
				labels = labels.cuda()

			appmodel = appmodel.train()

			output, _ = appmodel(trackframes, detectionframe)
			output = functional.log_softmax(output, dim=-1)
			loss = criterion(output, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			nn.utils.clip_grad_norm(appmodel.parameters(), 5.0)
						
			if((i+1)%10 == 0):
				print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}'.format( \
							epoch+1, num_epochs, i+1, train_data_size//train_batch_size, loss.data[0]))


		#Saving the model
		if(epoch%1 == 0):
			if not os.path.isdir(os.path.join(save_dir, "epoch{}".format(epoch))):
				os.makedirs(os.path.join(save_dir, "epoch{}".format(epoch)))
			filename = modeltype + '.pth'
			file = open(os.path.join(save_dir, "epoch{}".format(epoch), filename), 'wb')
			torch.save({'state_dict':appmodel.state_dict(), 'dict_args':dict_args}, file)
			print('Saving the model to {}'.format(save_dir+"epoch{}".format(epoch)))
			file.close()
		

		#Validation accuracy
		if(epoch%1 == 0): 
			num_correct = 0.0
			num_total = 0.0
			for j, batch in enumerate(val_dataloader):

				trackframes = Variable(torch.stack(batch[0]))
				detectionframe = Variable(torch.stack(batch[1]))
				labels = Variable(torch.LongTensor(batch[2]))

				if torch.cuda.is_available():
					trackframes = trackframes.cuda()
					detectionframe = detectionframe.cuda()
					labels = labels.cuda()

				appmodel = appmodel.eval()

				output, _ = appmodel(trackframes, detectionframe)

				predictions = output.max(dim=-1)[1]
				num_correct += (predictions == labels).sum().data[0]
				num_total += len(labels)

			accuracy = float(num_correct/num_total)
			print('Epoch {} Accuracy {} \n'.format(epoch, accuracy))
			valfile.write('Epoch {} Accuracy {} \n'.format(epoch, accuracy))

		#Saving the pickle
		if epoch == 0:
			if not use_pickle:
				pixel.save()

	valfile.close()

				
if __name__=='__main__':
	cuetype = sys.argv[1]
	use_pkl = sys.argv[2]
	if use_pkl == 'True': use_pickle = True
	else: use_pickle = False
	if cuetype == 'appearance':
		train_appearance(use_pickle)

