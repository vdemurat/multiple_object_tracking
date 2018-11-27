import os
import torch
import torch.utils.data as data

from dataclass import datastorage
from datautils import imagetotensor

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
		for frameid, coords in zip(tframeids, trackcoords):
			imagetensor = imagetotensor(self.pixel, self.pretrained, self.dataclass.data_folder, self.dataclass.dir_type, dirid, frameid, coords)
			trackframes.append(imagetensor)
		detectionframe = imagetotensor(self.pixel, self.pretrained, self.dataclass.data_folder, self.dataclass.dir_type, dirid, dframeid, detection)
		return [torch.stack(trackframes), detectionframe, label]

	def __len__(self):
		return len(self.data)

	def create(self):
		for items in self.dataclass.iterate(self.seqlen, self.mode):
			dirid, tframeids, trackcoords, dframeid, detection, label = items
			self.data.append([dirid, tframeids, trackcoords, dframeid, detection, label])

	def collate_fn(self, mini_batch):
		trackframesbatch, detectionframebatch, labelsbatch = zip(*mini_batch)
		return trackframesbatch, detectionframebatch, labelsbatch

if __name__ == '__main__':
	train_folder = 'MOT17/train/'
	datastorage = datastorage(train_folder)
	datastorage.prepare('DPM')
	datastorage.split(6)
	
	pixel = Pixel()
	train_dataset = dataset(datastorage, 6, pixel, 'train')
	train_dataset.create()

	val_dataset = dataset(datastorage, 6, pixel, 'val')
	val_dataset.create()

	trackframesbatch, detectionframebatch, labelsbatch = val_dataset.collate_fn([val_dataset[0], val_dataset[1]])
	print(trackframesbatch)
	print(detectionframebatch)
	print(labelsbatch)
