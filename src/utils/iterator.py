import random

import numpy as np

from utils.constants import *
from utils.eval import *


def build_tensor(batch, cuda):
	bsz = len(batch)
	max_ctx_ques_size = 0
	max_para_cnt = 0
	for data in batch:
		max_para_cnt = max(max_para_cnt, len(data['context_idxs']))
		assert len(data['ques_idxs']) <= QUES_LIMIT
		for para in data['context_idxs']:
			assert len(para) <= PARA_LIMIT
			max_ctx_ques_size = max(max_ctx_ques_size, 3 + len(para) + len(data['ques_idxs']))
	context_ques_idxs = torch.LongTensor(bsz, max_para_cnt, max_ctx_ques_size).fill_(UNK_IDX)
	context_ques_masks = torch.LongTensor(bsz, max_para_cnt, max_ctx_ques_size).fill_(0)
	context_ques_segments = torch.LongTensor(bsz, max_para_cnt, max_ctx_ques_size).fill_(1)
	answer_masks = torch.FloatTensor(bsz, max_para_cnt, max_ctx_ques_size).fill_(0.)
	ques_size = np.zeros(bsz, dtype=int)
	y1 = np.zeros((bsz, 2), dtype=int)
	y2 = np.zeros((bsz, 2), dtype=int)
	y1_flat = torch.LongTensor(bsz)
	y2_flat = torch.LongTensor(bsz)
	q_type = torch.LongTensor(bsz)

	for data_i, data in enumerate(batch):
		ques_size[data_i] = len(data['ques_idxs'])
		context_ques_idxs[data_i, :, 0: 1] = CLS_IDX
		context_ques_idxs[data_i, :, 1: 1 + len(data['ques_idxs'])] = data['ques_idxs']
		context_ques_idxs[data_i, :, 1 + len(data['ques_idxs']): 2 + len(data['ques_idxs'])] = SEP_IDX
		context_ques_segments[data_i, :, : 2 + len(data['ques_idxs'])] = 0
		for para_i, para in enumerate(data['context_idxs']):
			context_ques_idxs[data_i, para_i, 2 + len(data['ques_idxs']): 2 + len(data['ques_idxs']) + len(para)] = para
			context_ques_idxs[data_i, :, 2 + len(data['ques_idxs']) + len(para): 3 + len(data['ques_idxs']) + len(para)]\
				= SEP_IDX
			context_ques_masks[data_i, para_i, : 3 + len(data['ques_idxs']) + len(para)] = 1
			answer_masks[data_i, para_i, 2 + len(data['ques_idxs']): 2 + len(data['ques_idxs']) + len(para)] = 1.

		if batch[data_i][Y1_KEY][1] >= 0:
			# TODO: set y1, y2
			y1[data_i] = data[Y1_KEY]
			y2[data_i] = data[Y2_KEY]
			y1_flat[data_i] = data[Y1_KEY][0] * max_ctx_ques_size + 2 + len(data['ques_idxs']) + data[Y1_KEY][1]
			y2_flat[data_i] = data[Y1_KEY][0] * max_ctx_ques_size + 2 + len(data['ques_idxs']) + data[Y2_KEY][1]
			q_type[data_i] = 0
		elif batch[data_i][Y1_KEY][1] == -1:
			y1[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			y2[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			y1_flat[data_i] = IGNORE_INDEX
			y2_flat[data_i] = IGNORE_INDEX
			q_type[data_i] = 1
		elif batch[data_i][Y1_KEY][1] == -2:
			y1[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			y2[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			y1_flat[data_i] = IGNORE_INDEX
			y2_flat[data_i] = IGNORE_INDEX
			q_type[data_i] = 2
		else:
			assert False

	# TODO: start_mapping, end_mapping, is_support, orig_idxs
	if cuda:
		context_ques_idxs = context_ques_idxs.cuda()
		context_ques_masks = context_ques_masks.cuda()
		context_ques_segments = context_ques_segments.cuda()
		answer_masks = answer_masks.cuda()
		y1_flat = y1_flat.cuda()
		y2_flat = y2_flat.cuda()
		q_type = q_type.cuda()
	return context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, ques_size, q_type, y1, y2, y1_flat, y2_flat


class DataIterator(object):
	def __init__(self, datapoints, bsz, shuffle, debug=False):
		self.datapoints = datapoints
		self.bsz = bsz

		if shuffle:
			random.shuffle(self.datapoints)
		self.bkt_ptr = 0
		self.shuffle = shuffle
		self.debug = debug

	def __iter__(self):
		while True:
			start_id = self.bkt_ptr
			cur_bsz = min(self.bsz, len(self.datapoints) - start_id)

			cur_batch = self.datapoints[start_id: start_id + cur_bsz]

			ids = [data[ID_KEY] for data in cur_batch]
			context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, ques_size, q_type, y1, y2, y1_flat, y2_flat \
				= build_tensor(cur_batch, not self.debug)

			self.bkt_ptr += cur_bsz
			if self.bkt_ptr >= len(self.datapoints):
				break

			yield {
				FULL_BATCH_KEY: cur_batch,
				IDS_KEY: ids,
				CONTEXT_QUES_IDXS_KEY: context_ques_idxs,
				CONTEXT_QUES_MASKS_KEY: context_ques_masks,
				CONTEXT_QUES_SEGMENTS_KEY: context_ques_segments,
				ANSWER_MASKS_KEY: answer_masks,
				QUES_SIZE_KEY: ques_size,
				Q_TYPE_KEY: q_type,
				Y1_KEY: y1,
				Y2_KEY: y2,
				Y1_FLAT_KEY: y1_flat,
				Y2_FLAT_KEY: y2_flat,
			}
