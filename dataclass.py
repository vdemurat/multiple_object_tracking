import os
import random

import numpy as np

class datastorage():
	def __init__(self, data_folder):
		self.data_folder = data_folder
		self.tracks_data = {} #{dirindex : {trackindex : [{frameindex, detectionid}]}}
		self.detections = {} #{dirindex : {frameindex : {detectionid : (p1, p2, p3, p4)}}}
		self.test_data = {} #{dirindex : {frameindex : []}}
		self.traintracks_data = {}
		self.valtracks_data = {}
		self.negsamples = 1
		self.dir_type = None

	def generateid(self, findex, dirindex):
		if dirindex not in self.detections: return '0'
		if findex not in self.detections[dirindex]: return '0'
		return str(len(self.detections[dirindex][findex]))

	def split(self, seqlen):
		for dirindex, dirvalue in self.tracks_data.items():
			self.traintracks_data[dirindex] = {}
			self.valtracks_data[dirindex] = {}
			for trackindex, tracklist in self.tracks_data[dirindex].items():
				if len(tracklist) - seqlen > 12:
					splitindex = int((len(tracklist)-seqlen)*0.9)
					traintracklist = tracklist[:splitindex+seqlen]
					valtracklist = tracklist[splitindex+1:]
					self.traintracks_data[dirindex][trackindex] = traintracklist
					self.valtracks_data[dirindex][trackindex] = valtracklist
				else:
					self.traintracks_data[dirindex][trackindex] = tracklist
					self.valtracks_data[dirindex][trackindex] = []

	def prepare(self, dir_type):
		self.dir_type = dir_type
		for dir in os.listdir(self.data_folder):
			if dir_type in dir:
				groud_truth = open(os.path.join(self.data_folder, dir, 'gt/gt.txt'), 'r')
				dirindex = dir.split('-')[1]
				for line in groud_truth:
					frameid, trackid, p1, p2, p3, p4, flag, type, score = line.strip().split(',')
					detectionid = self.generateid(frameid, dirindex)

					if dirindex not in self.tracks_data:
						self.tracks_data[dirindex] = {}
					if trackid not in self.tracks_data[dirindex]:
						self.tracks_data[dirindex][trackid] = []
					self.tracks_data[dirindex][trackid].append({'fid': frameid, 'did': detectionid})

					if dirindex not in self.detections:
						self.detections[dirindex] = {}					
					if frameid not in self.detections[dirindex]:
						self.detections[dirindex][frameid] = {}
					self.detections[dirindex][frameid][detectionid] = (p1, p2, p3, p4)

	def print(self, dirindex, trackindex):
		for detection in self.tracks_data[dirindex][trackindex]:
			frameid, detectionid = detection['fid'], detection['did']
			coordinates = self.detections[dirindex][frameid][detectionid]
			print(frameid + ' -> ' + str(coordinates))

	def iterate(self, seqlen, mode='train'):
		if mode == 'train': self.itrtracks_data = self.traintracks_data
		else: self.itrtracks_data = self.valtracks_data

		for dirindex, dirvalue in self.itrtracks_data.items():
			for trackindex, tracklist in self.itrtracks_data[dirindex].items():
				if len(tracklist) > seqlen:
					for start in range(len(tracklist) - seqlen):
						tframeids = []
						trackcoords = []
						for seq in range(start, start+seqlen):
							frameid = self.itrtracks_data[dirindex][trackindex][seq]['fid']
							detectionid = self.itrtracks_data[dirindex][trackindex][seq]['did']
							coordinates = self.detections[dirindex][frameid][detectionid]
							tframeids.append(frameid)
							trackcoords.append(coordinates)
						dframeid = self.itrtracks_data[dirindex][trackindex][seq+1]['fid']
						detectionid = self.itrtracks_data[dirindex][trackindex][seq+1]['did']

						for sample in range(self.negsamples + 1):
							if sample == 0:
								detection = self.detections[dirindex][dframeid][detectionid]
								yield dirindex, tframeids, trackcoords, dframeid, detection, 1
							else:
								sampleid, sampledetection = random.choice(list(self.detections[dirindex][dframeid].items()))
								if sampleid != detectionid:
									yield dirindex, tframeids, trackcoords, dframeid, sampledetection, 0

	def preparetest(self, dir_type):
		for dir in os.listdir(self.data_folder):
			if dir_type in dir:
				boundingboxes = open(os.path.join(self.data_folder, dir, 'det/det.txt'), 'r')
				for line in boundingboxes:
					frameid, trackid, p1, p2, p3, p4, score, _, _, _ = line.strip().split(',')

					if dir not in self.test_data:
						self.test_data[dir] = {}					
					if frameid not in self.test_data[dir]:
						self.test_data[dir][frameid] = []

					self.test_data[dir][frameid].append((float(p1), float(p2), float(p3), float(p4)))

	def iteratetest(self):
		for dirindex, dirvalue in self.test_data.items():
			for frameindex, framelist in self.test_data[dirindex].items():
				yield dirindex, frameindex, framelist


if __name__ == '__main__':
	train_folder = 'MOT17/train/'
	data = datastorage(train_folder)
	'''data.prepare('DPM')
	data.split(6)
	#data.print('04', '141')
	count = 0
	for datum in data.iterate(6):
		print(datum)
		break
	for datum in data.iterate(6,'val'):
		print(datum)
		break'''

	data = datastorage(train_folder)
	data.preparetest('DPM')
	print(data.test_data['MOT17-02-DPM']['600'])

