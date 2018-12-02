import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from appearancemodel import AppearanceModel
from random import shuffle
from scipy.optimize import linear_sum_assignment

def load_model(modelpath, filename, modelclass):
	modeldict = open(os.path.join(modelpath, filename), 'rb')
	if torch.cuda.is_available():
		checkpoint = torch.load(modeldict)
	else:
		checkpoint = torch.load(modeldict, map_location=lambda storage, loc: storage)
	dict_args = checkpoint['dict_args']
	
	model = modelclass(dict_args)
	if torch.cuda.is_available():
		model = model.cuda()
	model.load_state_dict(checkpoint['state_dict'])
	return model


def hungarian(scores, num_trackings, random=False):
	#num_detections, num_tracks
	num_detections, num_tracks = scores.size()
	if random:
		indices = [i for i in range(num_detections)]
		shuffle(indices)
		if len(indices) > num_trackings:
			indices = indices[:num_trackings]
	else:
		scores = scores.permute(1, 0) #num_tracks, num_detections
		track_indices, detection_indices = linear_sum_assignment(scores.data.numpy())
		#print(track_indices)
		#print(detection_indices)
		count = 0
		indices = []
		for tind, dind in zip(track_indices, detection_indices):
			while(count < tind):
				indices.append(-1)
				count = count + 1
			indices.append(dind)
			count = count + 1

	for _ in range(num_trackings - len(indices)):
		indices.append(-1)

	#print(indices)
	#print("############")
	returnindices = torch.LongTensor(indices)
	if torch.cuda.is_available():
		returnindices = returnindices.cuda()
	return returnindices


def prepstepinput(input, indices, num_trackings):
	num_inputs, inp_dim = input.size()
	zerotensor = Variable(input.data.new(inp_dim).zero_())
	stepinputlist = []
	for index in indices:
		curindex = index.item()
		if curindex == -1: stepinputlist.append(zerotensor)
		else: stepinputlist.append(input[curindex])
	for _ in range(num_trackings - len(stepinputlist)):
		print("OK")
		stepinputlist.append(zerotensor)
	return torch.stack(stepinputlist)


if __name__=='__main__':
	model = load_model(os.getcwd(), 'appearance.pth', AppearanceModel)
	print(model.input_dim)