import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from appearancemodel import AppearanceModel

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

if __name__=='__main__':
	model = load_model(os.getcwd(), 'appearance.pth', AppearanceModel)
	print(model.input_dim)