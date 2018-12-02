import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from rnnlayer import RNNLayer, RNNTracker
from appearancemodel import AppearanceModel
from motionmodel import MotionModel

from trainutils import load_model, hungarian, prepstepinput

class TrackingModel(nn.Module):

	def __init__(	
					self, 
					dict_args, 
					use_appearance=False, 
					pretrained_appearance=True,
					sterile_appearance=False,
					use_motion=True, 
					pretrained_motion=True,
					sterile_motion=False,

				):
		
		super(TrackingModel, self).__init__()

		self.use_appearance = use_appearance
		self.use_motion = use_motion

		# Unnecessary repacking: can directly use dict_args
		self.app_input_dim = dict_args['app_input_dim']
		self.mot_input_dim = dict_args['mot_input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.feature_dim = dict_args['feature_dim']

		self.tracking_input_dim = 0

		if self.use_appearance:
			if pretrained_appearance:
				self.appearance_layer = load_model(os.getcwd(), 'appearance.pth', AppearanceModel)
				if sterile_appearance:
					self.appearance_layer.sterile()
			else:
				appearance_dict_args = {
						'input_dim' : self.app_input_dim,
						'rnn_hdim' : self.hidden_dim,
						'rnn_type' : self.rnn_type,
						'feature_dim' : self.feature_dim,			
				}
				self.appearance_layer = AppearanceModel(appearance_dict_args)

			self.tracking_input_dim += self.appearance_layer.feature_dim
		

		if self.use_motion:
			if pretrained_motion:
				self.motion_layer = load_model(os.getcwd(), 'motion.pth', MotionModel)
				if sterile_motion:
					self.motion_layer.sterile()
			else:
				motion_dict_args = {
						'input_dim' : self.mot_input_dim,
						'rnn_hdim' : self.hidden_dim,
						'rnn_type' : self.rnn_type,
						'feature_dim' : self.feature_dim,			
				}
				self.motion_layer = MotionModel(motion_dict_args)

			self.tracking_input_dim += self.motion_layer.feature_dim

		tracking_dict_args = {
					'input_dim' : self.tracking_input_dim,
					'rnn_hdim' : self.hidden_dim,
					'rnn_type' : self.rnn_type,
					'feature_dim' : self.feature_dim,
				}

		self.tracking_layer = RNNTracker(tracking_dict_args)


	def forward(self, isequence, itensor, msequence, mtensor):
		#isequence: batch_size * seq_len * inp_dim
		#itensor: batch_size * inp_dim

		if self.use_appearance:
			_, asequence = self.appearance_layer(isequence, itensor)
			csequence = asequence

		if self.use_motion:
			_, msequence = self.motion_layer(msequence, mtensor)
			if self.use_appearance:
				csequence = torch.cat((asequence, msequence), dim=2)
			else: csequence = msequence

		output = self.tracking_layer(csequence)

		return output


	def score(self, idetections, mdetections):
		#input: new_size * idim

		if self.use_appearance:
			num_inputs, inp_dim = idetections.size()
			scores = Variable(idetections.data.new(num_inputs, self.num_trackings).zero_())

		if self.use_motion:
			num_inputs, inp_dim = mdetections.size()
			scores = Variable(mdetections.data.new(num_inputs, self.num_trackings).zero_())

		for inp_step in range(num_inputs):

			inp_feat = None
			if self.use_appearance:
				input = idetections[inp_step].view(1, -1).repeat(self.num_trackings, 1)
				app_feat = self.appearance_layer.score(input)
				inp_feat = app_feat

			if self.use_motion:
				input = mdetections[inp_step].view(1, -1).repeat(self.num_trackings, 1)
				mot_feat = self.motion_layer.score(input)
				if self.use_appearance:
					inp_feat = torch.cat((app_feat, mot_feat), dim=-1)
				else: inp_feat = mot_feat

			score = self.tracking_layer.score(inp_feat)
			scores[inp_step] = score

		indices = hungarian(scores, self.num_trackings, random=False)
		cstepinput = None

		if self.use_appearance:
			istepinput = prepstepinput(idetections, indices, self.num_trackings)
			cistepinput = self.appearance_layer.score(istepinput)
			self.appearance_layer.step(istepinput)
			cstepinput = cistepinput
		if self.use_motion:
			mstepinput = prepstepinput(mdetections, indices, self.num_trackings)
			cmstepinput = self.motion_layer.score(mstepinput)
			if self.use_appearance:
				cstepinput = torch.cat((cistepinput, cmstepinput), dim=-1)
			else: cstepinput = cmstepinput
			self.motion_layer.step(mstepinput)

		self.step(cstepinput)
		return indices


	def step(self, input):
		self.tracking_layer.step(input)
		return

	def track(self, num_trackings=10):
		self.num_trackings = num_trackings
		self.tracking_layer.track(num_trackings)
		if self.use_appearance:
			self.appearance_layer.track(num_trackings)
		if self.use_motion:
			self.motion_layer.track(num_trackings)



if __name__=='__main__':

	dict_args = {
					'app_input_dim' : 4,
					'mot_input_dim' : 5,
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'feature_dim' : 5,
				}

	trackin_model = TrackingModel(
			dict_args,
			pretrained_appearance = False,
			pretrained_motion = False,
			use_appearance = True,
			use_motion = True
		)

	output = trackin_model(
		Variable(torch.randn(2,6,1000)), 
		Variable(torch.randn(2,1000)),
		Variable(torch.randn(2,6,5)), 
		Variable(torch.randn(2,5))
	)

	print(output.size())

	trackin_model.track(12)
	for i in range(4):
		scores = trackin_model.score(Variable(torch.randn(7-i,1000)), Variable(torch.randn(7-i,5)))
		print(scores)