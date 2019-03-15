import random

import numpy as np

from utils.eval import *
from utils.constants import *


def overlap_span(start, end, y1, y2):
	if start <= y1 < end or start <= y2 < end:
		return True
	return y1 <= start and y2 >= end


def sample_sent(batch, para_limit, char_limit, p=0.0, batch_p=None):
	new_batch = []
	for batch_i, data in enumerate(batch):
		length = len(data[START_END_FACTS_KEY])
		drop = np.random.rand(length) < (batch_p[batch_i][:length] if batch_p is not None else p)
		num_word_drop = 0
		context_idxs = data[CONTEXT_IDXS_KEY].data.new(para_limit).fill_(0)
		context_char_idxs = data[CONTEXT_CHAR_IDXS_KEY].data.new(para_limit, char_limit).fill_(0)
		y1 = data[Y1_KEY]
		y2 = data[Y2_KEY]
		y_offset = 0
		start_end_facts = []
		for j, cur_sp_dp in enumerate(data[START_END_FACTS_KEY]):
			if len(cur_sp_dp) == 3:
				start, end, is_sp_flag = tuple(cur_sp_dp)
				is_gold = None
			else:
				start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
			if start < end:
				if overlap_span(start, end, y1, y2):
					y_offset = num_word_drop
					y1 = data[Y1_KEY] - num_word_drop
					y2 = data[Y2_KEY] - num_word_drop
				if is_sp_flag or is_gold or overlap_span(start, end, data[Y1_KEY], data[Y2_KEY]) or not drop[j]:
					context_idxs[start - num_word_drop:end - num_word_drop] = data[CONTEXT_IDXS_KEY][start:end]
					context_char_idxs[start - num_word_drop:end - num_word_drop, :] \
						= data[CONTEXT_CHAR_IDXS_KEY][start:end]
					if is_gold is not None:
						start_end_facts.append((start - num_word_drop, end - num_word_drop, is_sp_flag, is_gold))
					else:
						start_end_facts.append((start - num_word_drop, end - num_word_drop, is_sp_flag))
				else:
					num_word_drop += (end - start)
		assert y1 < (context_idxs > 0).long().sum().item() and y2 < (context_idxs > 0).long().sum().item()
		new_batch.append({
			CONTEXT_IDXS_KEY: context_idxs,
			CONTEXT_CHAR_IDXS_KEY: context_char_idxs,
			QUES_IDXS_KEY: data[QUES_IDXS_KEY],
			QUES_CHAR_IDXS_KEY: data[QUES_CHAR_IDXS_KEY],
			Y1_KEY: y1,
			Y2_KEY: y2,
			Y_OFFSET_KEY: y_offset,
			ID_KEY: data[ID_KEY],
			START_END_FACTS_KEY: start_end_facts,
		})
	return new_batch


def build_ques_tensor(batch, char_limit, cuda):
	bsz = len(batch)
	max_q_len = max([(data[QUES_IDXS_KEY] > 0).long().sum().item() for data in batch])
	assert max_q_len > 0
	ques_idxs = torch.LongTensor(bsz, max_q_len)
	ques_char_idxs = torch.LongTensor(bsz, max_q_len, char_limit)
	if cuda:
		ques_idxs = ques_idxs.cuda()
		ques_char_idxs = ques_char_idxs.cuda()
	for i in range(len(batch)):
		ques_idxs[i].copy_(batch[i][QUES_IDXS_KEY][:max_q_len])
		ques_char_idxs[i].copy_(batch[i][QUES_CHAR_IDXS_KEY][:max_q_len])
	return ques_idxs, ques_char_idxs


def build_ctx_tensor(batch, sent_limit, char_limit, cuda):
	bsz = len(batch)
	max_c_len = max([(data[CONTEXT_IDXS_KEY] > 0).long().sum().item() for data in batch])
	max_sent_cnt = min(sent_limit, max([len(data[START_END_FACTS_KEY]) for data in batch]))
	assert max_c_len > 0 and max_sent_cnt > 0
	context_idxs = torch.LongTensor(bsz, max_c_len)
	context_char_idxs = torch.LongTensor(bsz, max_c_len, char_limit)
	context_lens = torch.LongTensor(bsz)
	start_mapping = torch.zeros(bsz, max_c_len, max_sent_cnt)
	end_mapping = torch.zeros(bsz, max_c_len, max_sent_cnt)
	all_mapping = torch.zeros(bsz, max_c_len, max_sent_cnt)
	is_support = torch.LongTensor(bsz, max_sent_cnt).fill_(IGNORE_INDEX)
	if cuda:
		context_idxs = context_idxs.cuda()
		context_char_idxs = context_char_idxs.cuda()
		context_lens = context_lens.cuda()
		start_mapping = start_mapping.cuda()
		end_mapping = end_mapping.cuda()
		all_mapping = all_mapping.cuda()
	for i in range(len(batch)):
		context_idxs[i].copy_(batch[i][CONTEXT_IDXS_KEY][:max_c_len])
		context_char_idxs[i].copy_(batch[i][CONTEXT_CHAR_IDXS_KEY][:max_c_len])
		context_lens[i] = (batch[i][CONTEXT_IDXS_KEY] > 0).long().sum()

		for j, cur_sp_dp in enumerate(batch[i][START_END_FACTS_KEY]):
			if j >= sent_limit: break
			if len(cur_sp_dp) == 3:
				start, end, is_sp_flag = tuple(cur_sp_dp)
			else:
				start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
			if start < end:
				start_mapping[i, start, j] = 1
				end_mapping[i, end - 1, j] = 1
				all_mapping[i, start:end, j] = 1
				is_support[i, j] = int(
					is_sp_flag or overlap_span(start, end, batch[i][Y1_KEY], batch[i][Y2_KEY]))
	return context_idxs, context_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, is_support


def build_ans_tensor(batch, cuda):
	bsz = len(batch)
	y1 = torch.LongTensor(bsz)
	y2 = torch.LongTensor(bsz)
	q_type = torch.LongTensor(bsz)
	y_offsets = np.zeros(bsz, dtype=int)
	for i in range(len(batch)):
		if batch[i][Y1_KEY] >= 0:
			y1[i] = batch[i][Y1_KEY]
			y2[i] = batch[i][Y2_KEY]
			q_type[i] = 0
		elif batch[i]['y1'] == -1:
			y1[i] = IGNORE_INDEX
			y2[i] = IGNORE_INDEX
			q_type[i] = 1
		elif batch[i][Y1_KEY] == -2:
			y1[i] = IGNORE_INDEX
			y2[i] = IGNORE_INDEX
			q_type[i] = 2
		elif batch[i][Y1_KEY] == -3:
			y1[i] = IGNORE_INDEX
			y2[i] = IGNORE_INDEX
			q_type[i] = 3
		else:
			assert False
		if Y_OFFSET_KEY in batch[i]:
			y_offsets[i] = batch[i][Y_OFFSET_KEY]
	if cuda:
		y1 = y1.cuda()
		y2 = y2.cuda()
		q_type = q_type.cuda()
	return y1, y2, q_type, y_offsets


class DataIterator(object):
	def __init__(self, buckets, bsz, para_limit, ques_limit, char_limit, shuffle, sent_limit, num_word, num_char,
	             debug=False, p=0.0):
		self.buckets = buckets
		self.bsz = bsz
		if para_limit is not None and ques_limit is not None:
			self.para_limit = para_limit
			self.ques_limit = ques_limit
		else:
			para_limit, ques_limit = 0, 0
			for bucket in buckets:
				for dp in bucket:
					para_limit = max(para_limit, dp[CONTEXT_IDXS_KEY].size(0))
					ques_limit = max(ques_limit, dp[QUES_IDXS_KEY].size(0))
			self.para_limit, self.ques_limit = para_limit, ques_limit
		self.char_limit = char_limit
		self.sent_limit = sent_limit
		self.num_word = num_word
		self.num_char = num_char

		self.num_buckets = len(self.buckets)
		self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
		if shuffle:
			for i in range(self.num_buckets):
				random.shuffle(self.buckets[i])
		self.bkt_ptrs = [0 for i in range(self.num_buckets)]
		self.shuffle = shuffle
		self.debug = debug
		self.p = p

	def __iter__(self):
		while True:
			if len(self.bkt_pool) == 0: break
			bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
			start_id = self.bkt_ptrs[bkt_id]
			cur_bucket = self.buckets[bkt_id]
			cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

			cur_batch = cur_bucket[start_id: start_id + cur_bsz]
			cur_batch = sample_sent(cur_batch, self.para_limit, self.char_limit, p=self.p)
			cur_batch.sort(key=lambda x: (x[CONTEXT_IDXS_KEY] > 0).long().sum(), reverse=True)
			full_batch = cur_batch

			ids = [data[ID_KEY] for data in cur_batch]
			context_idxs, context_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, is_support = \
				build_ctx_tensor(cur_batch, self.sent_limit, self.char_limit, not self.debug)
			ques_idxs, ques_char_idxs = build_ques_tensor(cur_batch, self.char_limit, not self.debug)
			_, _, _, y_offsets = build_ans_tensor(cur_batch, not self.debug)

			cur_batch = sample_sent(cur_batch, self.para_limit, self.char_limit, p=self.p)

			context_idxs_r, context_char_idxs_r, _, _, _, _, _ \
				= build_ctx_tensor(cur_batch, self.sent_limit, self.char_limit, not self.debug)
			y1_r, y2_r, q_type, _ = build_ans_tensor(cur_batch, not self.debug)

			self.bkt_ptrs[bkt_id] += cur_bsz
			if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
				self.bkt_pool.remove(bkt_id)

			yield {
				FULL_BATCH_KEY: full_batch,
				CONTEXT_IDXS_KEY: context_idxs.contiguous().clamp(0, self.num_word - 1),
				CONTEXT_IDXS_R_KEY: context_idxs_r.contiguous().clamp(0, self.num_word - 1),
				QUES_IDXS_KEY: ques_idxs.contiguous().clamp(0, self.num_word - 1),
				CONTEXT_CHAR_IDXS_KEY: context_char_idxs.contiguous().clamp(0, self.num_char - 1),
				CONTEXT_CHAR_IDXS_R_KEY: context_char_idxs_r.contiguous().clamp(0, self.num_char - 1),
				QUES_CHAR_IDXS_KEY: ques_char_idxs.contiguous().clamp(0, self.num_char - 1),
				CONTEXT_LENS_KEY: context_lens,
				IDS_KEY: ids,
				IS_SUPPORT_KEY: is_support.contiguous(),
				START_MAPPING_KEY: start_mapping,
				END_MAPPING_KEY: end_mapping,
				ALL_MAPPING_KEY: all_mapping,
				Y1_R_KEY: y1_r,
				Y2_R_KEY: y2_r,
				Y_OFFSETS_KEY: y_offsets,
				Q_TYPE_KEY: q_type,
			}
