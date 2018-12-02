import torch
import torch.utils.data as data

from dataset import dataset, testset
from dataclass import datastorage
from datautils import Pixel

def get_train_data(batch_size, seq_len, dir_type, pixel, pretrained=None, shuffle = True, num_workers = 0):
	train_folder = 'MOT17/train/'
	datastorageobject = datastorage(train_folder)
	datastorageobject.prepare(dir_type)
	datastorageobject.split(seq_len)
	
	traindatasetobject = dataset(datastorageobject, seq_len, pixel, pretrained, 'train')
	traindatasetobject.create()

	valdatasetobject = dataset(datastorageobject, seq_len, pixel, pretrained, 'val')
	valdatasetobject.create()

	traindataloader = data.DataLoader(traindatasetobject, batch_size=batch_size, collate_fn=traindatasetobject.collate_fn,  shuffle=shuffle, num_workers=num_workers)
	valdataloader = data.DataLoader(valdatasetobject, batch_size=batch_size, collate_fn=valdatasetobject.collate_fn,  shuffle=shuffle, num_workers=num_workers)
	return traindataloader, traindatasetobject.__len__(), valdataloader, valdatasetobject.__len__()


def get_eval_data(batch_size, dir_type, pixel, pretrained=None):
	eval_folder = 'MOT17/train/'
	datastorageobject = datastorage(eval_folder)
	datastorageobject.preparetest(dir_type)

	evaldatasetobject = testset(datastorageobject, pixel, pretrained)
	evaldatasetobject.create()

	evaldataloader = data.DataLoader(evaldatasetobject, batch_size=batch_size, collate_fn=evaldatasetobject.collate_fn,  shuffle=False)
	return evaldataloader, evaldatasetobject.__len__()


if __name__ == '__main__':
	#pixel = Pixel()
	'''train_dataloader, train_datalen, val_dataloader, val_datalen = get_train_data(2, 6, 'DPM', pixel)
	for batch in train_dataloader:
		print(batch[0][0].size())
		print(batch[1][0].size())
		print(batch[2])
		break
	print(train_datalen)

	for batch in val_dataloader:
		print(batch[0][0].size())
		print(batch[1][0].size())
		print(batch[2])
		break
	print(val_datalen)'''

	eval_dataloader, eval_datalen = get_eval_data(64, 'DPM', None)
	for batch in eval_dataloader:
		print(batch[0][0].size())
		print(batch[1])
		print(batch[2])


