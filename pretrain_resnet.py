import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from datautils import Pixel
from pretrain_utils import dataset, get_train_data, ResNetPretraining

use_pickle = False

num_epochs = 300
learning_rate = 0.1
train_batch_size = 1      # tried 64 originally, but memory errors if batch_size >= 4...
sequence_length = 1
directory_type = 'DPM'
modeltype = 'resnet_pretrained'

pixel = Pixel('pixel.pkl')
if use_pickle: pixel.load()

print('Building the train loader...')
train_dataloader, train_data_size, val_dataloader, val_data_size = \
        get_train_data(train_batch_size, sequence_length, directory_type, pixel)

save_dir = 'models/{}/'.format(modeltype)
valfile = open('val.txt', 'a')


dict_args = {'input_dim' : 500, 'feature_dim' : 100}

resnet = ResNetPretraining(dict_args)
criterion = nn.NLLLoss() 
optimizer = optim.Adadelta(resnet.parameters(), lr=learning_rate, rho=0.95, eps=1e-06, weight_decay=0)

if torch.cuda.is_available():
    resnet = resnet.cuda()
    criterion = criterion.cuda()

print('Entering training loop...')
for epoch in range(num_epochs):
    resnet = resnet.train()
    print('Epoch change...')
    for i, batch in enumerate(train_dataloader):
        trackframes = Variable(torch.stack(batch[0]).view(train_batch_size,3,224,224))
        detectionframe = Variable(torch.stack(batch[1]).view(train_batch_size,3,224,224))
        labels = Variable(torch.LongTensor(batch[2]))

        if torch.cuda.is_available():
            trackframes = trackframes.cuda()
            detectionframe = detectionframe.cuda()
            labels = labels.cuda()

        output = resnet(trackframes, detectionframe)
        output = functional.log_softmax(output, dim=-1)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm(resnet.parameters(), 5.0)

        if(i%10 == 0):
            print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}'.format( \
                        epoch+1, num_epochs, i+1, train_data_size//train_batch_size, loss.data[0]))


    #Saving the model
    if(epoch%1 == 0):
        if not os.path.isdir(os.path.join(save_dir, "epoch{}".format(epoch))):
            os.makedirs(os.path.join(save_dir, "epoch{}".format(epoch)))
        filename = modeltype + '.pth'
        file = open(os.path.join(save_dir, "epoch{}".format(epoch), filename), 'wb')
        torch.save({'state_dict':resnet.state_dict(), 'dict_args':dict_args}, file)
        print('Saving the model to {}'.format(save_dir+"epoch{}".format(epoch)))
        file.close()

    #Validation accuracy
    resnet = resnet.eval()
    if(epoch%1 == 0): 
        num_correct = 0.0
        num_total = 0.0
        for j, batch in enumerate(val_dataloader):
            trackframes = Variable(torch.stack(batch[0]).view(train_batch_size,3,224,224))
            detectionframe = Variable(torch.stack(batch[1]).view(train_batch_size,3,224,224))
            labels = Variable(torch.LongTensor(batch[2]))

            if torch.cuda.is_available():
                trackframes = trackframes.cuda()
                detectionframe = detectionframe.cuda()
                labels = labels.cuda()

            output = resnet(trackframes, detectionframe)

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
