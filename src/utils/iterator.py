import random
from copy import deepcopy

import numpy as np

from utils.constants import *
from utils.eval import *


def build_compact_tensor(batch, cuda, para_idxs):
	bsz = len(batch)
	compact_ctx_ques_sizes = []
	compact_max_sent_cnt = 0
	for data_i, data in enumerate(batch):
		assert len(data[QUES_IDXS_KEY]) <= QUES_LIMIT
		cur_compact_ctx_ques_size = len(data[QUES_IDXS_KEY]) + 3 + \
									sum([len(data[CONTEXT_IDXS_KEY][i]) for i in para_idxs[data_i]])
		compact_ctx_ques_sizes.append(cur_compact_ctx_ques_size)
		compact_max_sent_cnt = max(compact_max_sent_cnt,
								   sum([len(data[START_END_FACTS_KEY][i]) for i in para_idxs[data_i]]))
	compact_max_ctx_ques_size = max(compact_ctx_ques_sizes)
	assert compact_max_ctx_ques_size <= BERT_LIMIT

	compact_context_ques_idxs = torch.LongTensor(bsz, 1, compact_max_ctx_ques_size).fill_(UNK_IDX)
	compact_context_ques_masks = torch.zeros(bsz, 1, compact_max_ctx_ques_size)
	compact_context_ques_segments = torch.LongTensor(bsz, 1, compact_max_ctx_ques_size).fill_(1)
	compact_all_mapping = torch.zeros(bsz, 1, compact_max_sent_cnt, compact_max_ctx_ques_size)
	compact_is_support = torch.LongTensor(bsz, 1, compact_max_sent_cnt).fill_(IGNORE_INDEX)
	compact_answer_masks = torch.zeros(bsz, 1, compact_max_ctx_ques_size)
	# make sure not two entries pointing to the same positions
	compact_to_orig_mapping = {
		'token': np.full((bsz, compact_max_ctx_ques_size, 2), INVALID_INDEX, dtype=int),
		'sent': np.full((bsz, compact_max_sent_cnt, 2), INVALID_INDEX, dtype=int)
	}

	for data_i, data in enumerate(batch):
		compact_context_ques_idxs[data_i, :, 0: 1] = CLS_IDX
		compact_context_ques_idxs[data_i, :, 1: 1 + len(data[QUES_IDXS_KEY])] = data[QUES_IDXS_KEY]
		compact_context_ques_idxs[data_i, :, 1 + len(data[QUES_IDXS_KEY]): 2 + len(data[QUES_IDXS_KEY])] = SEP_IDX
		compact_context_ques_segments[data_i, :, : 2 + len(data[QUES_IDXS_KEY])] = 0
		compact_context_ques_masks[data_i, :, : compact_ctx_ques_sizes[data_i]] = 0
		compact_answer_masks[data_i, :, 2 + len(data[QUES_IDXS_KEY]): compact_ctx_ques_sizes[data_i] - 1] = 1.

		token_offset = 2 + len(data[QUES_IDXS_KEY])
		sent_offset = 0
		for para_i in para_idxs[data_i]:
			para_ctx_size = len(data[CONTEXT_IDXS_KEY][para_i])
			compact_context_ques_idxs[data_i, :, token_offset: para_ctx_size + token_offset] \
				= data[CONTEXT_IDXS_KEY][para_i]

			for sent_i, sent in enumerate(data[START_END_FACTS_KEY][para_i]):
				raw_start, raw_end, is_sp = sent
				compact_is_support[data_i, :, sent_i + sent_offset] = int(is_sp)
				compact_all_mapping[data_i, :, sent_i + sent_offset, raw_start + token_offset: raw_end + token_offset] \
					= 1.
				compact_to_orig_mapping['sent'][data_i, sent_i + sent_offset, :] = [para_i, sent_i]
			compact_to_orig_mapping['token'][data_i, token_offset: token_offset + para_ctx_size, 0] = para_i
			compact_to_orig_mapping['token'][data_i, token_offset: token_offset + para_ctx_size, 1] \
				= np.arange(para_ctx_size)

			token_offset += para_ctx_size
			sent_offset += len(data[START_END_FACTS_KEY][para_i])

		compact_context_ques_idxs[data_i, :, compact_ctx_ques_sizes[data_i] - 1: compact_ctx_ques_sizes[data_i]] \
			= SEP_IDX

	if cuda:
		compact_context_ques_idxs = compact_context_ques_idxs.cuda()
		compact_context_ques_masks = compact_context_ques_masks.cuda()
		compact_context_ques_segments = compact_context_ques_segments.cuda()
		compact_answer_masks = compact_answer_masks.cuda()
		compact_all_mapping = compact_all_mapping.cuda()
		compact_is_support = compact_is_support.cuda()

	return compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, \
		   compact_answer_masks, compact_is_support, compact_all_mapping, compact_to_orig_mapping


def build_compact_tensor_no_support(batch, para_idxs, cuda):
	compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, compact_answer_masks, _, compact_all_mapping, compact_to_orig_mapping \
		= build_compact_tensor(batch, cuda, para_idxs)
	return compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, \
		   compact_answer_masks, compact_all_mapping, compact_to_orig_mapping


def get_mixed_para_idxs(batch, compact_para_cnt):
	pure_para_idxs = [[i for i, has_sp_fact in enumerate(data[HAS_SP_KEY]) if has_sp_fact] for data in batch]
	left_out_para_idxs = [[i for i, has_sp_fact in enumerate(data[HAS_SP_KEY]) if not has_sp_fact] for data in batch]
	mixed_para_idxs = []
	for pure_para_idx, left_out_para_idx, data in zip(pure_para_idxs, left_out_para_idxs, batch):
		ctx_ques_size = 3 + len(data[QUES_IDXS_KEY]) + sum([len(data[CONTEXT_IDXS_KEY][i]) for i in pure_para_idx])
		mixed_para_idx = deepcopy(pure_para_idx)
		for i in left_out_para_idx:
			if len(mixed_para_idx) >= compact_para_cnt:
				break
			if ctx_ques_size + len(data[CONTEXT_IDXS_KEY][i]) <= BERT_LIMIT:
				mixed_para_idx.append(i)
				ctx_ques_size += len(data[CONTEXT_IDXS_KEY][i])
		random.shuffle(mixed_para_idx)
		mixed_para_idxs.append(mixed_para_idx)
	return mixed_para_idxs


def build_tensor(batch, compact_para_cnt, cuda):
	bsz = len(batch)
	# indices of all paragraphs that are fed into reasoner
	para_idxs = get_mixed_para_idxs(batch, compact_para_cnt)
	max_ctx_ques_size = 0
	max_para_cnt = 0
	# max number of sentences per paragraph
	max_sent_cnt = 0
	for data_i, data in enumerate(batch):
		assert len(data[QUES_IDXS_KEY]) <= QUES_LIMIT
		max_para_cnt = max(max_para_cnt, len(data[CONTEXT_IDXS_KEY]))
		for para_i, para in enumerate(data[CONTEXT_IDXS_KEY]):
			assert len(para) <= PARA_LIMIT
			max_ctx_ques_size = max(max_ctx_ques_size, 3 + len(para) + len(data[QUES_IDXS_KEY]))
			max_sent_cnt = max(max_sent_cnt, len(data[START_END_FACTS_KEY][para_i]))

	context_ques_idxs = torch.LongTensor(bsz, max_para_cnt, max_ctx_ques_size).fill_(UNK_IDX)
	context_ques_masks = torch.zeros(bsz, max_para_cnt, max_ctx_ques_size)
	context_ques_segments = torch.LongTensor(bsz, max_para_cnt, max_ctx_ques_size).fill_(1)
	is_support = torch.LongTensor(bsz, max_para_cnt, max_sent_cnt).fill_(IGNORE_INDEX)
	has_support = torch.LongTensor(bsz, max_para_cnt).fill_(IGNORE_INDEX)
	y1 = np.zeros((bsz, 2), dtype=int)
	y2 = np.zeros((bsz, 2), dtype=int)
	compact_y1 = torch.LongTensor(bsz).fill_(IGNORE_INDEX)
	compact_y2 = torch.LongTensor(bsz).fill_(IGNORE_INDEX)
	q_type = torch.LongTensor(bsz)

	for data_i, data in enumerate(batch):
		context_ques_idxs[data_i, :, 0: 1] = CLS_IDX
		context_ques_idxs[data_i, :, 1: 1 + len(data[QUES_IDXS_KEY])] = data[QUES_IDXS_KEY]
		context_ques_idxs[data_i, :, 1 + len(data[QUES_IDXS_KEY]): 2 + len(data[QUES_IDXS_KEY])] = SEP_IDX
		context_ques_segments[data_i, :, : 2 + len(data[QUES_IDXS_KEY])] = 0

		for para_i, para in enumerate(data[CONTEXT_IDXS_KEY]):
			context_ques_idxs[data_i, para_i, 2 + len(data[QUES_IDXS_KEY]): 2 + len(data[QUES_IDXS_KEY]) + len(para)] \
				= para
			context_ques_idxs[data_i, :, 2 + len(data[QUES_IDXS_KEY]) + len(para):
										 3 + len(data[QUES_IDXS_KEY]) + len(para)] = SEP_IDX
			context_ques_masks[data_i, para_i, : 3 + len(data[QUES_IDXS_KEY]) + len(para)] = 1
			has_support[data_i, para_i] = int(data[HAS_SP_KEY][para_i])

			for sent_i, sent in enumerate(data[START_END_FACTS_KEY][para_i]):
				raw_start, raw_end, is_sp = sent
				is_support[data_i, para_i, sent_i] = int(is_sp)

		# build answer span and question type
		token_offset = 2 + len(data[QUES_IDXS_KEY])
		sent_offset = 0
		for compact_para_i, para_i in enumerate(para_idxs[data_i]):
			para_ctx_size = len(data[CONTEXT_IDXS_KEY][para_i])
			if data[Y1_KEY][0] == para_i and data[Y1_KEY][1] >= 0:
				compact_y1[data_i] = token_offset + data[Y1_KEY][1]
				compact_y2[data_i] = token_offset + data[Y2_KEY][1]
			token_offset += para_ctx_size
			sent_offset += len(data[START_END_FACTS_KEY][para_i])

		if data[Y1_KEY][1] >= 0:
			y1[data_i] = data[Y1_KEY]
			y2[data_i] = data[Y2_KEY]
			q_type[data_i] = 0
		elif data[Y1_KEY][1] == -1:
			y1[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			y2[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			q_type[data_i] = 1
		elif data[Y1_KEY][1] == -2:
			y1[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			y2[data_i] = (IGNORE_INDEX, IGNORE_INDEX)
			q_type[data_i] = 2
		else:
			assert False

	# TODO: start_mapping, end_mapping, is_support, orig_idxs
	if cuda:
		context_ques_idxs = context_ques_idxs.cuda()
		context_ques_masks = context_ques_masks.cuda()
		context_ques_segments = context_ques_segments.cuda()
		is_support = is_support.cuda()
		has_support = has_support.cuda()
		compact_y1 = compact_y1.cuda()
		compact_y2 = compact_y2.cuda()
		q_type = q_type.cuda()

	compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, compact_answer_masks, compact_is_support, compact_all_mapping, compact_to_orig_mapping \
		= build_compact_tensor(batch, cuda, para_idxs)

	return context_ques_idxs, compact_context_ques_idxs, \
		   context_ques_masks, compact_context_ques_masks, \
		   context_ques_segments, compact_context_ques_segments, \
		   compact_answer_masks, \
		   is_support, compact_is_support, has_support, \
		   compact_all_mapping, \
		   compact_y1, compact_y2, q_type, y1, y2, \
		   compact_to_orig_mapping


class DataIterator(object):
	def __init__(self, datapoints, bsz, shuffle, compact_para_cnt, debug=False):
		self.datapoints = datapoints
		self.bsz = bsz

		if shuffle:
			random.shuffle(self.datapoints)
		self.bkt_ptr = 0
		self.shuffle = shuffle
		self.compact_para_cnt = compact_para_cnt
		self.debug = debug

	def __iter__(self):
		while True:
			start_id = self.bkt_ptr
			cur_bsz = min(self.bsz, len(self.datapoints) - start_id)

			cur_batch = self.datapoints[start_id: start_id + cur_bsz]

			ids = [data[ID_KEY] for data in cur_batch]
			context_ques_idxs, compact_context_ques_idxs, \
			context_ques_masks, compact_context_ques_masks, \
			context_ques_segments, compact_context_ques_segments, \
			compact_answer_masks, \
			is_support, compact_is_support, has_support, \
			compact_all_mapping, \
			compact_y1, compact_y2, q_type, y1, y2, compact_to_orig_mapping \
				= build_tensor(cur_batch, self.compact_para_cnt, not self.debug)

			self.bkt_ptr += cur_bsz
			if self.bkt_ptr >= len(self.datapoints):
				break

			yield {
				FULL_BATCH_KEY: cur_batch,
				IDS_KEY: ids,
				CONTEXT_QUES_IDXS_KEY: context_ques_idxs,
				COMPACT_CONTEXT_QUES_IDXS_KEY: compact_context_ques_idxs,
				CONTEXT_QUES_MASKS_KEY: context_ques_masks,
				COMPACT_CONTEXT_QUES_MASKS_KEY: compact_context_ques_masks,
				CONTEXT_QUES_SEGMENTS_KEY: context_ques_segments,
				COMPACT_CONTEXT_QUES_SEGMENTS_KEY: compact_context_ques_segments,
				COMPACT_ANSWER_MASKS_KEY: compact_answer_masks,
				Q_TYPE_KEY: q_type,
				IS_SUPPORT_KEY: is_support,
				COMPACT_IS_SUPPORT_KEY: compact_is_support,
				HAS_SP_KEY: has_support,
				COMPACT_ALL_MAPPING_KEY: compact_all_mapping,
				COMPACT_Y1_KEY: compact_y1,
				COMPACT_Y2_KEY: compact_y2,
				Y1_KEY: y1,
				Y2_KEY: y2,
				COMPACT_TO_ORIG_MAPPING_KEY: compact_to_orig_mapping,
			}
