import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim

from trackingmodel import TrackingModel
from datautils import PreTrainedResnet, Pixel

from dataloader import get_eval_data
from trainutils import load_model

def eval_tracker(use_pickle=False):

	trackingmodel = load_model(os.getcwd(), 'tracker.pth', TrackingModel)

	trackingmodel.eval()

	eval_batch_size = 1
	directory_type = 'DPM'

	eval_folder = 'MOT17/eval/'

	pixel = Pixel('pixel.pkl')
	if use_pickle: pixel.load()
	pretrained = PreTrainedResnet({'intermediate_layers':['fc']})
	if torch.cuda.is_available():
		pretrained = pretrained.cuda()

	eval_dataloader, eval_data_size = \
		get_eval_data(eval_batch_size, directory_type, pixel, pretrained)

	trackingmodel.track(30)
	prev_dir = None
	tracks = []

	def savetracks(tracks, pdir):
		file = open(os.path.join(eval_folder, pdir+'.txt'), 'w')
		nptracks = np.stack(tracks) #num_frames * num_tracks * 4
		nptracks = np.moveaxis(nptracks, 0, 1) #num_tracks * num_frames * 4
		num_tracks, num_frames, num_coords = nptracks.shape
		for track in range(num_tracks):
			for frame in range(num_frames):
				if nptracks[track][frame][0] != -1:
					file.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1\n'.format(
						track+1,
						frame+1,
						nptracks[track][frame][0], 
						nptracks[track][frame][1],
						nptracks[track][frame][2],
						nptracks[track][frame][3]
						)
					)
		file.close()


	for i, batch in enumerate(eval_dataloader):
		detections = Variable(torch.stack(batch[0]))
		frameid = batch[1]
		dirtype = batch[2][0]
		frametensors = Variable(torch.stack(batch[3]))

		if torch.cuda.is_available():
			detections = detections.cuda()
			frametensors = frametensors.cuda()

		if prev_dir != dirtype and prev_dir != None:
			print(prev_dir)
			savetracks(tracks, prev_dir)
			tracks = []

		detections = detections.squeeze(0)
		indices = trackingmodel.score(frametensors, detections)

		frametracks = []
		for index in indices:
			if index != -1:
				frametracks.append(detections[index].data.numpy())
			else:
				frametracks.append(np.asarray([-1, -1, -1, -1]))
		tracks.append(np.stack(frametracks))

		prev_dir = dirtype

	print(prev_dir)
	savetracks(tracks, prev_dir)


if __name__=='__main__':
	eval_tracker()