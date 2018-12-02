import torch
import torch.nn as nn
from torch.autograd import Variable

from rnnlayer import RNNLayer

class MotionModel(nn.Module):

	def __init__(self, dict_args):
		super(MotionModel, self).__init__()

		# Unnecessary repacking: can directly use dict_args
		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.feature_dim = dict_args['feature_dim']

		dict_args = {
					'input_dim' : self.input_dim,
					'rnn_hdim' : self.hidden_dim,
					'rnn_type' : self.rnn_type,
					'feature_dim' : self.feature_dim,
				}

		self.motion_layer = RNNLayer(dict_args)

	def sterile(self):
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, isequence, itensor):
		#isequence: batch_size * seq_len * inp_dim
		#itensor: batch_size * inp_dim

		output, features = self.motion_layer(isequence, itensor)
		return output, features

	def track(self, num_trackings):
		self.motion_layer.track(num_trackings)
	
	def score(self, detection):
		features = self.motion_layer.score(detection)
		return features

	def step(self, input):
		self.motion_layer.step(input)

if __name__=='__main__':
	dict_args = {
					'input_dim' : 4,
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'feature_dim' : 5,
				}

	motlayer = MotionModel(dict_args)
	output, features = motlayer(
		Variable(torch.randn(2,4,4)), 
		Variable(torch.randn(2,4))
	)
	print(features.size())

	file = open('motion.pth', 'wb')
	torch.save({'state_dict':motlayer.state_dict(), 'dict_args':dict_args}, file)

