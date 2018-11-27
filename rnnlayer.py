import torch
import torch.nn as nn
from torch.autograd import Variable


class RNNLayer(nn.Module):

	def __init__(self, dict_args):
		super(RNNLayer, self).__init__()

		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.feature_dim = dict_args['feature_dim']

		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim) #ToDO
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRUCell(self.input_dim, self.hidden_dim)
		elif self.rnn_type == 'RNN':
			pass

		self.features = nn.Linear(self.input_dim+self.hidden_dim, self.feature_dim)
		self.linear = nn.Linear(self.feature_dim, 2)

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			c_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			return c_0
		elif self.rnn_type == 'GRU':
			pass
		elif self.rnn_type == 'RNN':
			pass

	def forward(self, isequence, itensor, hidden_t = None):
		#isequence: batch_size * seq_len * inp_dim
		#hidden_t: batch_size * hidden_dim
		#itensor: batch_size * inp_dim

		batch_size, num_frames, _ = isequence.size()
		isequence = isequence.permute(1,0,2) #isequence: seq_len * batch_size * inp_dim

		if hidden_t is not None: h_t = hidden_t
		else: h_t = Variable(next(self.parameters()).data.new(batch_size, self.hidden_dim).zero_())
		if self.rnn_type == 'LSTM': c_t = self.init_hidden(batch_size)

		osequence = Variable(isequence.data.new(num_frames, batch_size, self.feature_dim).zero_())

		for step in range(num_frames):
			input = isequence[step]
			if self.rnn_type == 'LSTM':
				h_t, c_t = self.rnn(input, (h_t, c_t)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_t = self.rnn(input, h_t) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

			if step != num_frames-1:
				osequence[step] = self.features(torch.cat((h_t, isequence[step+1]), dim=-1)) #batch_size*feature_dim

		osequence[step] = self.features(torch.cat((h_t, itensor), dim=-1)) #batch_size*feature_dim

		output = self.linear(osequence[step])

		osequence = osequence.permute(1,0,2)

		return output, osequence #batch_size*2, batch_size * seq_len * feature_dim


class RNNTracker(nn.Module):
	def __init__(self, dict_args):
		super(RNNTracker, self).__init__()

		self.input_dim = dict_args['input_dim']
		self.hidden_dim = dict_args['rnn_hdim']
		self.rnn_type = dict_args['rnn_type']
		self.feature_dim = dict_args['feature_dim']

		if self.rnn_type == 'LSTM':
			self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim) #ToDO
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRUCell(self.input_dim, self.hidden_dim)
		elif self.rnn_type == 'RNN':
			pass

		self.features = nn.Linear(self.hidden_dim, self.feature_dim)
		self.linear = nn.Linear(self.feature_dim, 2)

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			c_0 = Variable(weight.new(batch_size, self.hidden_dim).zero_())
			return c_0
		elif self.rnn_type == 'GRU':
			pass
		elif self.rnn_type == 'RNN':
			pass

	def forward(self, isequence, hidden_t = None):
		#isequence: batch_size * seq_len * inp_dim
		#hidden_t: batch_size * hidden_dim

		batch_size, num_frames, _ = isequence.size()
		isequence = isequence.permute(1,0,2) #isequence: seq_len * batch_size * inp_dim

		if hidden_t is not None: h_t = hidden_t
		else: h_t = Variable(next(self.parameters()).data.new(batch_size, self.hidden_dim).zero_())
		if self.rnn_type == 'LSTM': c_t = self.init_hidden(batch_size)

		for step in range(num_frames):
			input = isequence[step]
			if self.rnn_type == 'LSTM':
				h_t, c_t = self.rnn(input, (h_t, c_t)) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'GRU':
				h_t = self.rnn(input, h_t) #h_t: batch_size*hidden_dim
			elif self.rnn_type == 'RNN':
				pass

		features = self.features(h_t) #batch_size*feature_dim
		output = self.linear(features)

		return output #batch_size*2


if __name__=='__main__':
	dict_args = {
					'input_dim' : 4,
					'rnn_hdim' : 3,
					'rnn_type' : 'LSTM',
					'feature_dim' : 5,
				}

	rnnlayer = RNNLayer(dict_args)
	output, features = rnnlayer(
		Variable(torch.randn(2,4,4)), 
		Variable(torch.randn(2,4))
	)
	print(output.size())
	print(features.size())
