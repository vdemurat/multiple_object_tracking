import os
import torch
import time
import pickle

import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision
import torchvision.models as models
from torch.autograd import Variable
torchvision.set_image_backend('accimage')

class PreTrainedResnet(nn.Module):

	def __init__(self, dict_args):
		super(PreTrainedResnet, self).__init__()

		self.intermediate_layers = dict_args['intermediate_layers']
		self.pretrained_model = models.resnet18(pretrained=True).eval()
		if torch.cuda.is_available():
			self.pretrained_model = self.pretrained_model.cuda()
		for param in self.pretrained_model.parameters():
			param.requires_grad = False

	def forward(self, x):
		intermediate_features = []
		for name, module in self.pretrained_model._modules.items():
			if name=='fc':
				x = x.squeeze().contiguous()
			x = module(x)
			if name in self.intermediate_layers:
				intermediate_features += [x]
		return intermediate_features[0]

class Pixel():
	def __init__(self, pklfilepath):
		self.pathtoimage = {}
		self.pathtotensor = {}
		self.pklfilepath = pklfilepath

	def addimage(self, path, image):
		self.pathtoimage[path] = image

	def getimage(self, path):
		if path in self.pathtoimage:
			return self.pathtoimage[path]
		return None

	def addtensor(self, path, tensor):
		self.pathtotensor[path] = tensor

	def gettensor(self, path):
		if path in self.pathtotensor:
			return self.pathtotensor[path]
		return None

	def save(self):
		pklfile = open(self.pklfilepath, 'wb')
		pickle.dump(self.pathtotensor, pklfile)
		pklfile.close()
		print("saving the pklfile to {0}".format(self.pklfilepath))

	def load(self):
		pklfile = open(self.pklfilepath, 'rb')
		self.pathtotensor = pickle.load(pklfile)
		pklfile.close()


def imagetotensor(pixel, pretrained, datafolder, dirtype, dirid, frameid, coordinates):
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
		tensor = pretrained(Variable(image)).view(-1).data.cpu()
		pixel.addtensor(pixelkey, tensor)
	return tensor

def coordstotensor(coordinates):
	top_x, top_y, height, width = coordinates
	return torch.FloatTensor([float(top_x), float(top_y), float(height), float(width)])

if __name__=='__main__':
	datafolder = 'MOT17/train/'
	dirtype = 'DPM'
	dirid = '02'
	frameid = '1'
	coords = (940.04,436.58,27.284,83.853)
	tensor = imagetotensor(PreTrainedResnet({'intermediate_layers':['fc']}), datafolder, dirtype, dirid, frameid, coords)
	print(tensor.size())
