import random

import numpy as np

from util import *


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

	def init_batch(self):
		context_idxs = torch.LongTensor(self.bsz, self.para_limit)
		ques_idxs = torch.LongTensor(self.bsz, self.ques_limit)
		context_char_idxs = torch.LongTensor(self.bsz, self.para_limit, self.char_limit)
		ques_char_idxs = torch.LongTensor(self.bsz, self.ques_limit, self.char_limit)
		y1 = torch.LongTensor(self.bsz)
		y2 = torch.LongTensor(self.bsz)
		q_type = torch.LongTensor(self.bsz)
		start_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit)
		end_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit)
		all_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit)
		is_support = torch.LongTensor(self.bsz, self.sent_limit)
		if not self.debug:
			context_idxs = context_idxs.cuda()
			ques_idxs = ques_idxs.cuda()
			context_char_idxs = context_char_idxs.cuda()
			ques_char_idxs = ques_char_idxs.cuda()
			y1 = y1.cuda()
			y2 = y2.cuda()
			q_type = q_type.cuda()
			start_mapping = start_mapping.cuda()
			end_mapping = end_mapping.cuda()
			all_mapping = all_mapping.cuda()
			is_support = is_support.cuda()
		return all_mapping, context_char_idxs, context_idxs, end_mapping, is_support, q_type, ques_char_idxs, ques_idxs, start_mapping, y1, y2

	def sample_sent(self, batch, p=0.0):
		assert self.para_limit > 0 and self.char_limit > 0
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
		all_mapping, context_char_idxs_l, context_idxs_l, end_mapping_l, is_support, _, ques_char_idxs, ques_idxs, start_mapping_l, _, _ = self.init_batch()
		_, context_char_idxs_r, context_idxs_r, end_mapping_r, _, q_type, _, _, start_mapping_r, y1, y2 = self.init_batch()

		while True:
			if len(self.bkt_pool) == 0: break
			bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
			start_id = self.bkt_ptrs[bkt_id]
			cur_bucket = self.buckets[bkt_id]
			cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

			ids = []
			y_offsets_l, y_offsets_r = [], []
			max_sent_cnt_l, max_sent_cnt_r = 0, 0
			for mapping in [start_mapping_l, end_mapping_l, start_mapping_r, end_mapping_r, all_mapping]:
				mapping.zero_()
			is_support.fill_(IGNORE_INDEX)

			cur_batch = cur_bucket[start_id: start_id + cur_bsz]
			cur_batch = self.sample_sent(cur_batch, p=self.p)
			cur_batch.sort(key=lambda x: (x['context_idxs'] > 0).long().sum(), reverse=True)

			for i in range(len(cur_batch)):
				context_idxs_l[i].copy_(cur_batch[i]['context_idxs'])
				ques_idxs[i].copy_(cur_batch[i]['ques_idxs'])
				context_char_idxs_l[i].copy_(cur_batch[i]['context_char_idxs'])
				ques_char_idxs[i].copy_(cur_batch[i]['ques_char_idxs'])
				ids.append(cur_batch[i]['id'])
				y_offsets_l.append(cur_batch[i]['y_offset'])

				for j, cur_sp_dp in enumerate(cur_batch[i]['start_end_facts']):
					if j >= self.sent_limit: break
					if len(cur_sp_dp) == 3:
						start, end, is_sp_flag = tuple(cur_sp_dp)
					else:
						start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
					if start < end:
						start_mapping_l[i, start, j] = 1
						end_mapping_l[i, end - 1, j] = 1
						all_mapping[i, start:end, j] = 1
						is_support[i, j] = int(
							is_sp_flag or overlap_span(start, end, cur_batch[i]['y1'], cur_batch[i]['y2']))

				max_sent_cnt_l = max(max_sent_cnt_l, len(cur_batch[i]['start_end_facts']))

			input_lengths_l = (context_idxs_l[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len_l = int(input_lengths_l.max())
			max_q_len = int((ques_idxs[:cur_bsz] > 0).long().sum(dim=1).max())

			cur_batch = self.sample_sent(cur_batch, p=self.p)

			for i in range(len(cur_batch)):
				context_idxs_r[i].copy_(cur_batch[i]['context_idxs'])
				context_char_idxs_r[i].copy_(cur_batch[i]['context_char_idxs'])
				if cur_batch[i]['y1'] >= 0:
					y1[i] = cur_batch[i]['y1']
					y2[i] = cur_batch[i]['y2']
					q_type[i] = 0
				elif cur_batch[i]['y1'] == -1:
					y1[i] = IGNORE_INDEX
					y2[i] = IGNORE_INDEX
					q_type[i] = 1
				elif cur_batch[i]['y1'] == -2:
					y1[i] = IGNORE_INDEX
					y2[i] = IGNORE_INDEX
					q_type[i] = 2
				elif cur_batch[i]['y1'] == -3:
					y1[i] = IGNORE_INDEX
					y2[i] = IGNORE_INDEX
					q_type[i] = 3
				else:
					assert False
				y_offsets_r.append(cur_batch[i]['y_offset'])

				for j, cur_sp_dp in enumerate(cur_batch[i]['start_end_facts']):
					if j >= self.sent_limit: break
					if len(cur_sp_dp) == 3:
						start, end, is_sp_flag = tuple(cur_sp_dp)
					else:
						start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
					if start < end:
						start_mapping_r[i, start, j] = 1
						end_mapping_r[i, end - 1, j] = 1

				max_sent_cnt_r = max(max_sent_cnt_r, len(cur_batch[i]['start_end_facts']))

			max_c_len_r = int((context_idxs_r[:cur_bsz] > 0).long().sum(dim=1).max())

			self.bkt_ptrs[bkt_id] += cur_bsz
			if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
				self.bkt_pool.remove(bkt_id)

			yield {
				'context_idxs_l': context_idxs_l[:cur_bsz, :max_c_len_l].contiguous().clamp(0, self.num_word - 1),
				'context_idxs_r': context_idxs_r[:cur_bsz, :max_c_len_r].contiguous().clamp(0, self.num_word - 1),
				'ques_idxs': ques_idxs[:cur_bsz, :max_q_len].contiguous().clamp(0, self.num_word - 1),
				'context_char_idxs_l': context_char_idxs_l[:cur_bsz, :max_c_len_l].contiguous().clamp(0,
				                                                                                      self.num_char - 1),
				'context_char_idxs_r': context_char_idxs_r[:cur_bsz, :max_c_len_r].contiguous().clamp(0,
				                                                                                      self.num_char - 1),
				'ques_char_idxs': ques_char_idxs[:cur_bsz, :max_q_len].contiguous().clamp(0, self.num_char - 1),
				'context_lens_l': input_lengths_l,
				'ids': ids,
				'is_support': is_support[:cur_bsz, :max_sent_cnt_l].contiguous(),
				'start_mapping_l': start_mapping_l[:cur_bsz, :max_c_len_l, :max_sent_cnt_l],
				'start_mapping_r': start_mapping_r[:cur_bsz, :max_c_len_r, :max_sent_cnt_r],
				'end_mapping_l': end_mapping_l[:cur_bsz, :max_c_len_l, :max_sent_cnt_l],
				'end_mapping_r': end_mapping_r[:cur_bsz, :max_c_len_r, :max_sent_cnt_r],
				'all_mapping': all_mapping[:cur_bsz, :max_c_len_l, :max_sent_cnt_l],
				'y1': y1[:cur_bsz],
				'y2': y2[:cur_bsz],
				'y_offsets_l': np.array(y_offsets_l),
				'y_offsets_r': np.array(y_offsets_r),
				'q_type': q_type[:cur_bsz],
			}
