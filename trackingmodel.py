import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from rnnlayer import RNNLayer, RNNTracker
from appearancemodel import AppearanceModel

from trainutils import load_model

class TrackingModel(nn.Module):

	def __init__(	
					self, 
					dict_args, 
					use_appearance=True, 
					pretrained_appearance=True,
					sterile_appearance=False,
					use_motion=False, 
					pretrained_motion=False,
					sterile_motion=False,

				):
		
		super(TrackingModel, self).__init__()

		self.use_appearance = use_appearance
		self.use_motion = use_motion

		# Unnecessary repacking: can directly use dict_args
		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.feature_dim = dict_args['feature_dim']

		if pretrained_appearance:
			self.appearance_layer = load_model(os.getcwd(), 'appearance.pth', AppearanceModel)
			if sterile_appearance:
				self.appearance_layer.sterile()
		else:
			appearance_dict_args = {
					'input_dim' : self.input_dim,
					'rnn_hdim' : self.hidden_dim,
					'rnn_type' : self.rnn_type,
					'feature_dim' : self.feature_dim,			
			}
			self.appearance_layer = RNNLayer(appearance_dict_args)

		self.tracking_input_dim = 0
		if self.use_appearance: 
			self.tracking_input_dim += self.appearance_layer.feature_dim
		if self.use_motion:
			self.tracking_input_dim += self.motion_layer.feature_dim

		tracking_dict_args = {
					'input_dim' : self.tracking_input_dim,
					'rnn_hdim' : self.hidden_dim,
					'rnn_type' : self.rnn_type,
					'feature_dim' : self.feature_dim,
				}

		self.tracking_layer = RNNTracker(tracking_dict_args)


	def forward(self, isequence, itensor):
		#isequence: batch_size * seq_len * inp_dim
		#itensor: batch_size * inp_dim

		if self.use_appearance:
			_, asequence = self.appearance_layer(isequence, itensor)
			csequence = asequence

		if self.use_motion:
			_, msequence = self.motion_layer(isequence, itensor)
			if self.use_appearance:
				csequence = torch.cat((asequence, msequence), dim=2)
			else: csequence = msequence

		output = self.tracking_layer(csequence)

		return output

if __name__=='__main__':

	dict_args = {
					'input_dim' : 4,
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'feature_dim' : 5,
				}

	trackin_model = TrackingModel(
			dict_args,
			pretrained_appearance = False,
		)
	print(next(trackin_model.appearance_layer.parameters()).requires_grad)
	
	trackin_model = TrackingModel(
			dict_args,
			pretrained_appearance = True,
		)
	print(next(trackin_model.appearance_layer.parameters()).requires_grad)

	output = trackin_model(
		Variable(torch.randn(2,6,1000)), 
		Variable(torch.randn(2,1000))
	)

	print(output.size())
