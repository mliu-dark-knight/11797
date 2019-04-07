import random

from utils.constants import *
from utils.eval import *


def build_tensor(batch, cuda):
	bsz = len(batch)
	max_ctx_ques_size = 0
	max_para_cnt = 0
	for data in batch:
		max_para_cnt = max(max_para_cnt, len(data['context_idxs']))
		for para in zip(data['context_idxs']):
			max_ctx_ques_size = max(max_ctx_ques_size, len(para) + len(data['ques_idxs'] + 3))

	context_ques_idxs = torch.LongTensor((bsz, max_para_cnt, max_ctx_ques_size)).fill_(UNK_IDX)
	context_ques_masks = torch.LongTensor((bsz, max_para_cnt, max_ctx_ques_size)).fill_(0)
	context_ques_segments = torch.LongTensor((bsz, max_para_cnt, max_ctx_ques_size)).fill_(1)
	q_type = torch.LongTensor(bsz)

	for data_i, data in enumerate(batch):
		context_ques_idxs[data_i, :, 0: 1] = CLS_IDX
		context_ques_idxs[data_i, :, 1: 1 + len(data['ques_idxs'])] = data['ques_idxs']
		context_ques_idxs[data_i, :, 1 + len(data['ques_idxs']): 2 + len(data['ques_idxs'])] = SEP_IDX
		context_ques_segments[data_i, :, : 2 + len(data['ques_idxs'])] = 0
		for para_i, para in enumerate(data['context_idxs']):
			context_ques_idxs[data_i, para_i, 2 + len(data['ques_idxs']): 3 + len(data['ques_idxs']) + len(para)] = para
			context_ques_masks[data_i, para_i, : 4 + len(data['ques_idxs']) + len(para)] = 1
		context_ques_idxs[data_i, :,
		3 + len(data['ques_idxs']) + len(para): 4 + len(data['ques_idxs']) + len(para)] = SEP_IDX

		if batch[data_i][Y1_KEY][1] >= 0:
			# TODO: set y1, y2
			q_type[data_i] = 0
		elif batch[data_i][Y1_KEY][1] == -1:
			q_type[data_i] = 1
		elif batch[data_i][Y1_KEY][1] == -2:
			q_type[data_i] = 2
		else:
			assert False

	# TODO: start_mapping, end_mapping, is_support, orig_idxs
	if cuda:
		context_ques_idxs = context_ques_idxs.cuda()
		context_ques_masks = context_ques_masks.cuda()
		context_ques_segments = context_ques_segments.cuda()
		q_type = q_type.cuda()
	return context_ques_idxs, context_ques_masks, context_ques_segments, q_type


class DataIterator(object):
	def __init__(self, bucket, bsz, shuffle, debug=False):
		self.bucket = bucket
		self.bsz = bsz

		if shuffle:
			random.shuffle(self.bucket)
		self.bkt_ptr = 0
		self.shuffle = shuffle
		self.debug = debug

	def __iter__(self):
		while True:
			start_id = self.bkt_ptr
			cur_bsz = min(self.bsz, len(self.bucket) - start_id)

			cur_batch = self.bucket[start_id: start_id + cur_bsz]

			ids = [data[ID_KEY] for data in cur_batch]
			context_ques_idxs, context_ques_masks, context_ques_segments, q_type = build_tensor(cur_batch,
																								not self.debug)

			self.bkt_ptr += cur_bsz
			if self.bkt_ptr >= len(self.bucket):
				break

			yield {
				FULL_BATCH_KEY: cur_batch,
				IDS_KEY: ids,
				CONTEXT_QUES_IDXS_KEY: context_ques_idxs,
				CONTEXT_QUES_MASKS_KEY: context_ques_masks,
				CONTEXT_QUES_SEGMENTS_KEY: context_ques_segments,
				Q_TYPE_KEY: q_type,
			}
