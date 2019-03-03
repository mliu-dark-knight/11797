import random

import numpy as np

from utils.eval import *


def overlap_span(start, end, y1, y2):
	if start <= y1 < end or start <= y2 < end:
		return True
	return y1 <= start and y2 >= end


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
					para_limit = max(para_limit, dp['context_idxs'].size(0))
					ques_limit = max(ques_limit, dp['ques_idxs'].size(0))
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

	def build_ques_tensor(self, batch):
		bsz = len(batch)
		max_q_len = max([(data['ques_idxs'] > 0).long().sum().item() for data in batch])
		assert max_q_len > 0
		ques_idxs = torch.LongTensor(bsz, max_q_len)
		ques_char_idxs = torch.LongTensor(bsz, max_q_len, self.char_limit)
		if not self.debug:
			ques_idxs = ques_idxs.cuda()
			ques_char_idxs = ques_char_idxs.cuda()
		for i in range(len(batch)):
			ques_idxs[i].copy_(batch[i]['ques_idxs'][:max_q_len])
			ques_char_idxs[i].copy_(batch[i]['ques_char_idxs'][:max_q_len])
		return ques_idxs, ques_char_idxs

	def build_ctx_tensor(self, batch):
		bsz = len(batch)
		max_c_len = max([(data['context_idxs'] > 0).long().sum().item() for data in batch])
		max_sent_cnt = min(self.sent_limit, max([len(data['start_end_facts']) for data in batch]))
		assert max_c_len > 0 and max_sent_cnt > 0
		context_idxs = torch.LongTensor(bsz, max_c_len)
		context_char_idxs = torch.LongTensor(bsz, max_c_len, self.char_limit)
		context_lens = torch.LongTensor(bsz)
		start_mapping = torch.zeros(bsz, max_c_len, max_sent_cnt)
		end_mapping = torch.zeros(bsz, max_c_len, max_sent_cnt)
		all_mapping = torch.zeros(bsz, max_c_len, max_sent_cnt)
		is_support = torch.LongTensor(bsz, max_sent_cnt).fill_(IGNORE_INDEX)
		if not self.debug:
			context_idxs = context_idxs.cuda()
			context_char_idxs = context_char_idxs.cuda()
			context_lens = context_lens.cuda()
			start_mapping = start_mapping.cuda()
			end_mapping = end_mapping.cuda()
			all_mapping = all_mapping.cuda()
		for i in range(len(batch)):
			context_idxs[i].copy_(batch[i]['context_idxs'][:max_c_len])
			context_char_idxs[i].copy_(batch[i]['context_char_idxs'][:max_c_len])
			context_lens[i] = (batch[i]['context_idxs'] > 0).long().sum()

			for j, cur_sp_dp in enumerate(batch[i]['start_end_facts']):
				if j >= self.sent_limit: break
				if len(cur_sp_dp) == 3:
					start, end, is_sp_flag = tuple(cur_sp_dp)
				else:
					start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
				if start < end:
					start_mapping[i, start, j] = 1
					end_mapping[i, end - 1, j] = 1
					all_mapping[i, start:end, j] = 1
					is_support[i, j] = int(
						is_sp_flag or overlap_span(start, end, batch[i]['y1'], batch[i]['y2']))
		return context_idxs, context_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, is_support

	def build_ans_tensor(self, batch):
		bsz = len(batch)
		y1 = torch.LongTensor(bsz)
		y2 = torch.LongTensor(bsz)
		q_type = torch.LongTensor(bsz)
		y_offsets = np.zeros(bsz, dtype=int)
		for i in range(len(batch)):
			if batch[i]['y1'] >= 0:
				y1[i] = batch[i]['y1']
				y2[i] = batch[i]['y2']
				q_type[i] = 0
			elif batch[i]['y1'] == -1:
				y1[i] = IGNORE_INDEX
				y2[i] = IGNORE_INDEX
				q_type[i] = 1
			elif batch[i]['y1'] == -2:
				y1[i] = IGNORE_INDEX
				y2[i] = IGNORE_INDEX
				q_type[i] = 2
			elif batch[i]['y1'] == -3:
				y1[i] = IGNORE_INDEX
				y2[i] = IGNORE_INDEX
				q_type[i] = 3
			else:
				assert False
			y_offsets[i] = batch[i]['y_offset']
		return y1, y2, q_type, y_offsets

	def sample_sent(self, batch, p=0.0):
		new_batch = []
		for data in batch:
			drop = np.random.rand(len(data['start_end_facts'])) < p
			num_word_drop = 0
			context_idxs = data['context_idxs'].data.new(self.para_limit).fill_(0)
			context_char_idxs = data['context_char_idxs'].data.new(self.para_limit, self.char_limit).fill_(0)
			y1 = data['y1']
			y2 = data['y2']
			y_offset = 0
			start_end_facts = []
			for j, cur_sp_dp in enumerate(data['start_end_facts']):
				if len(cur_sp_dp) == 3:
					start, end, is_sp_flag = tuple(cur_sp_dp)
					is_gold = None
				else:
					start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
				if start < end:
					if overlap_span(start, end, y1, y2):
						y_offset = num_word_drop
						y1 = data['y1'] - num_word_drop
						y2 = data['y2'] - num_word_drop
					if is_sp_flag or is_gold or overlap_span(start, end, data['y1'], data['y2']) or not drop[j]:
						context_idxs[start - num_word_drop:end - num_word_drop] = data['context_idxs'][start:end]
						context_char_idxs[start - num_word_drop:end - num_word_drop, :] = data['context_char_idxs'][
						                                                                  start:end]
						if is_gold is not None:
							start_end_facts.append((start - num_word_drop, end - num_word_drop, is_sp_flag, is_gold))
						else:
							start_end_facts.append((start - num_word_drop, end - num_word_drop, is_sp_flag))
					else:
						num_word_drop += (end - start)
			assert y1 < (context_idxs > 0).long().sum().item() and y2 < (context_idxs > 0).long().sum().item()
			new_batch.append({
				'context_idxs': context_idxs,
				'context_char_idxs': context_char_idxs,
				'ques_idxs': data['ques_idxs'],
				'ques_char_idxs': data['ques_char_idxs'],
				'y1': y1,
				'y2': y2,
				'y_offset': y_offset,
				'id': data['id'],
				'start_end_facts': start_end_facts,
			})
		return new_batch

	def __iter__(self):
		while True:
			if len(self.bkt_pool) == 0: break
			bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
			start_id = self.bkt_ptrs[bkt_id]
			cur_bucket = self.buckets[bkt_id]
			cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

			cur_batch = cur_bucket[start_id: start_id + cur_bsz]
			cur_batch = self.sample_sent(cur_batch, p=self.p)
			cur_batch.sort(key=lambda x: (x['context_idxs'] > 0).long().sum(), reverse=True)

			ids = [data['id'] for data in cur_batch]
			context_idxs, context_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, is_support = \
				self.build_ctx_tensor(cur_batch)
			ques_idxs, ques_char_idxs = self.build_ques_tensor(cur_batch)
			y1, y2, _, y_offsets = self.build_ans_tensor(cur_batch)

			cur_batch = self.sample_sent(cur_batch, p=self.p)

			context_idxs_r, context_char_idxs_r, _, _, _, _, _ = self.build_ctx_tensor(cur_batch)
			y1_r, y2_r, q_type, y_offsets_r = self.build_ans_tensor(cur_batch)
			y_offsets_r = y_offsets + y_offsets_r

			self.bkt_ptrs[bkt_id] += cur_bsz
			if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
				self.bkt_pool.remove(bkt_id)

			yield {
				'context_idxs': context_idxs.contiguous().clamp(0, self.num_word - 1),
				'context_idxs_r': context_idxs_r.contiguous().clamp(0, self.num_word - 1),
				'ques_idxs': ques_idxs.contiguous().clamp(0, self.num_word - 1),
				'context_char_idxs': context_char_idxs.contiguous().clamp(0, self.num_char - 1),
				'context_char_idxs_r': context_char_idxs_r.contiguous().clamp(0, self.num_char - 1),
				'ques_char_idxs': ques_char_idxs.contiguous().clamp(0, self.num_char - 1),
				'context_lens': context_lens,
				'ids': ids,
				'is_support': is_support.contiguous(),
				'start_mapping': start_mapping,
				'end_mapping': end_mapping,
				'all_mapping': all_mapping,
				'y1': y1,
				'y1_r': y1_r,
				'y2': y2,
				'y2_r': y2_r,
				'y_offsets': y_offsets,
				'y_offsets_r': y_offsets_r,
				'q_type': q_type,
			}
