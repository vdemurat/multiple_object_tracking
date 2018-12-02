import os
import sys
import time
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as functional
import torch.optim as optim

import torchvision
import torchvision.models as models
from torchvision import transforms

from dataclass import datastorage
from datautils import Pixel, coordstotensor




class dataset(data.Dataset):
	def __init__(self, dataclass, seqlen, pixel, mode='train'):
		self.dataclass = dataclass
		self.seqlen = seqlen
		self.data = []
		
		self.pixel = pixel
		self.mode = mode

	def __getitem__(self, index):
		dirid, tframeids, trackcoords, dframeid, detection, label = self.data[index]
		trackframes = []
		trackcoordslist = []
		for frameid, coords in zip(tframeids, trackcoords):
			imagetensor = imagetotensor(self.pixel, self.dataclass.data_folder, self.dataclass.dir_type, dirid, frameid, coords)
			trackframes.append(imagetensor)
			coordstensor = coordstotensor(coords)
			trackcoordslist.append(coordstensor)
		detectionframe = imagetotensor(self.pixel, self.dataclass.data_folder, self.dataclass.dir_type, dirid, dframeid, detection)
		detectiontensor = coordstotensor(detection)
        
		return [torch.stack(trackframes), detectionframe, label, torch.stack(trackcoordslist), detectiontensor]

	def __len__(self):
		return len(self.data)

	def create(self):
		for items in self.dataclass.iterate(self.seqlen, self.mode):
			dirid, tframeids, trackcoords, dframeid, detection, label = items
			self.data.append([dirid, tframeids, trackcoords, dframeid, detection, label])

	def collate_fn(self, mini_batch):
		trackframesbatch, detectionframebatch, labelsbatch, trackcoordsbatch, detectioncoordbatch = zip(*mini_batch)
		return trackframesbatch, detectionframebatch, labelsbatch, trackcoordsbatch, detectioncoordbatch
    
    
    
def get_train_data(batch_size, seq_len, dir_type, pixel,shuffle = True, num_workers = 0):
	train_folder = '/scratch/vdm245/tracking/train/'
	datastorageobject = datastorage(train_folder)
	datastorageobject.prepare(dir_type)
	datastorageobject.split(seq_len)
	
	traindatasetobject = dataset(datastorageobject, seq_len, pixel, 'train')
	traindatasetobject.create()

	valdatasetobject = dataset(datastorageobject, seq_len, pixel, 'val')
	valdatasetobject.create()

	traindataloader = data.DataLoader(traindatasetobject, batch_size=batch_size, collate_fn=traindatasetobject.collate_fn,  shuffle=shuffle, num_workers=num_workers)
	valdataloader = data.DataLoader(valdatasetobject, batch_size=batch_size, collate_fn=valdatasetobject.collate_fn,  shuffle=shuffle, num_workers=num_workers)
	return traindataloader, traindatasetobject.__len__(), valdataloader, valdatasetobject.__len__()




def imagetotensor(pixel, datafolder, dirtype, dirid, frameid, coordinates):
	dirfolder = 'MOT17-' + dirid + '-' + dirtype
	filename = ''.join(['0' for _ in range(6 - len(frameid))]) + frameid

	pixelkey = dirfolder + '_' + filename + '_' + '_'.join(coordinates)
	tensor = pixel.gettensor(pixelkey)
	if tensor is None:
		imagekey = dirfolder + '_' + filename
		image = pixel.getimage(imagekey)

		if image is None:
			imagepath = os.path.join(datafolder, dirfolder, 'img1', filename + '.jpg')
			image = Image.open(imagepath).convert('RGB')
			pixel.addimage(imagekey, image)

		image = image.crop(
			(float(coordinates[0]), 
			float(coordinates[1]), 
			float(coordinates[0]) + float(coordinates[2]), 
			float(coordinates[1]) + float(coordinates[3]))
		)
			
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
		transform = transforms.Compose([
        		transforms.Resize(256),
        		#transforms.Scale(256),
        		transforms.CenterCrop(224),
        		transforms.ToTensor(),
        		normalize,
    		])
		image = transform(image).unsqueeze(0)
		if torch.cuda.is_available(): image = image.cuda()
		tensor = Variable(image).data.cpu()
		pixel.addtensor(pixelkey, tensor)
	return tensor




class ResNetPretraining(nn.Module):

    def __init__(self, dict_args):
        super(ResNetPretraining, self).__init__()
        self.input_dim = dict_args['input_dim']
        self.feature_dim = dict_args['feature_dim']
        
        self.resnet = models.resnet18(pretrained=True)
        self.dimred_layer = nn.Linear(1000, self.input_dim)
        self.features = nn.Linear(2 * self.input_dim, self.feature_dim)
        self.linear = nn.Linear(self.feature_dim, 2)
        
        
    def forward(self, itensor_1, itensor_2):
        encode_1 = self.dimred_layer(self.resnet(itensor_1))
        encode_2 = self.dimred_layer(self.resnet(itensor_2))
        out = self.linear(self.features(torch.cat((encode_1, encode_2), dim=-1)))
        return out
        