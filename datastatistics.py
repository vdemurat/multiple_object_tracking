import os
import numpy as np

def numtrackings(train_folder):
	for dir in os.listdir(train_folder):
		detector_types = ['DPM', 'FRCNN', 'SDP']
		for dtype in detector_types:
			if dtype in dir:
				groud_truth = open(os.path.join(train_folder, dir, 'gt/gt.txt'), 'r')
				track_count = 0
				for line in groud_truth:
					track_count = line.strip().split(',')[1]
				print(dir + ': ' + dtype + ': ' + track_count)

def tracklenghts(train_folder):
	def reindentify(track_indices):
		for i, tindex in enumerate(track_indices):
			if int(i+1) != int(tindex):
				print(i+1, tindex)
				print("Re-Identification Spotted")

	def tlengths(starts, ends):
		startsnp = np.asarray(starts)
		endsnp = np.asarray(ends)
		lengths = endsnp - startsnp
		print(lengths.mean())

	for dir in os.listdir(train_folder):
		detector_types = ['DPM']
		for dtype in detector_types:
			if 'DPM' in dir:
				groud_truth = open(os.path.join(train_folder, dir, 'gt/gt.txt'), 'r')
				track_starts = []
				track_ends = []
				track_indices = []
				prev_track = 0
				prev_end = -1
				for line in groud_truth:
					tokens = line.strip().split(',')
					cur_track = tokens[1]
					if cur_track != prev_track:
						track_starts.append(int(tokens[0]))
						track_ends.append(int(prev_end))
						track_indices.append(cur_track)
					prev_track = cur_track
					prev_end = tokens[0]
				track_ends.append(int(prev_end))
				track_ends = track_ends[1:]
				print(dir + ' ' + dtype)
				tlengths(track_starts, track_ends)
				reindentify(track_indices)
				print("###########################")

if __name__ == '__main__':
	train_folder = 'MOT17/test/'
	numtrackings(train_folder)
	tracklenghts(train_folder)
