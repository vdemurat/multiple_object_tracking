import torch
import torch.nn as nn
from torch.autograd import Variable

from rnnlayer import RNNLayer

class AppearanceModel(nn.Module):

	def __init__(self, dict_args):
		super(AppearanceModel, self).__init__()

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

		self.appearance_layer = RNNLayer(dict_args)

	def sterile(self):
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, isequence, itensor):
		#isequence: batch_size * seq_len * inp_dim
		#itensor: batch_size * inp_dim
		output, features = self.appearance_layer(isequence, itensor)
		return output, features


if __name__=='__main__':
	dict_args = {
					'input_dim' : 4,
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'feature_dim' : 5,
				}

	applayer = AppearanceModel(dict_args)
	output, features = applayer(
		Variable(torch.randn(2,4,4)), 
		Variable(torch.randn(2,4))
	)
	print(features.size())
	for param in applayer.parameters():
		print(param.requires_grad)
		break

	applayer.sterile()
	for param in applayer.parameters():
		print(param.requires_grad)
		break

	file = open('appearance.pth', 'wb')
	torch.save({'state_dict':applayer.state_dict(), 'dict_args':dict_args}, file)
	
	