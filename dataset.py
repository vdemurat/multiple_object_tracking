import os
import time

import torch
import torch.utils.data as data
from torch.autograd import Variable

from dataclass import datastorage
from datautils import imagetotensor, coordstotensor
from datautils import Pixel, PreTrainedResnet

class dataset(data.Dataset):
	def __init__(self, dataclass, seqlen, pixel, pretrained, mode='train'):
		self.dataclass = dataclass
		self.seqlen = seqlen
		self.data = []
		self.pretrained = pretrained
		self.pixel = pixel
		self.mode = mode

	def __getitem__(self, index):
		dirid, tframeids, trackcoords, dframeid, detection, label = self.data[index]
		trackframes = []
		trackcoordslist = []
		for frameid, coords in zip(tframeids, trackcoords):
			imagetensor = imagetotensor(self.pixel, self.pretrained, self.dataclass.data_folder, self.dataclass.dir_type, dirid, frameid, coords)
			trackframes.append(imagetensor)
			coordstensor = coordstotensor(coords)
			trackcoordslist.append(coordstensor)
		detectionframe = imagetotensor(self.pixel, self.pretrained, self.dataclass.data_folder, self.dataclass.dir_type, dirid, dframeid, detection)
		detectiontensor = coordstotensor(detection)

		'''batchframes = torch.stack(trackframes + [detectionframe])	
		batchtensors = self.pretrained(Variable(batchframes))
		trackframestensors = batchtensors[:-1].data.cpu()
		detectionframetensor = batchtensors[-1].view(-1).data.cpu()		
		return [trackframestensors, detectionframetensor, label]'''
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

if __name__ == '__main__':
	train_folder = 'MOT17/train/'
	datastorage = datastorage(train_folder)
	datastorage.prepare('DPM')
	datastorage.split(6)
	
	pixel = Pixel('pixel.pkl')
	pretrained = PreTrainedResnet({'intermediate_layers':['fc']})
	train_dataset = dataset(datastorage, 6, pixel, pretrained, 'train')
	train_dataset.create()

	val_dataset = dataset(datastorage, 6, pixel, pretrained, 'val')
	val_dataset.create()

	trackframesbatch, detectionframebatch, labelsbatch, trackcoordsbatch, detectionbatch = val_dataset.collate_fn([val_dataset[0], val_dataset[1]])
	print(trackframesbatch)
	print(detectionframebatch)
	print(labelsbatch)
	print(trackcoordsbatch)
	print(detectionbatch)
