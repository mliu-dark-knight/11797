import random

import spacy
import torch
import ujson as json
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.constants import *

nlp = spacy.blank("en")

import bisect
import re


def find_nearest(a, target, test_func=lambda x: True):
	idx = bisect.bisect_left(a, target)
	if (0 <= idx < len(a)) and a[idx] == target:
		return target, 0
	elif idx == 0:
		return a[0], abs(a[0] - target)
	elif idx == len(a):
		return a[-1], abs(a[-1] - target)
	else:
		d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
		d2 = abs(a[idx - 1] - target) if test_func(a[idx - 1]) else 1e200
		if d1 > d2:
			return a[idx - 1], d2
		else:
			return a[idx], d1


def fix_span(para, offsets, span):
	# span not necessarily in para
	span = span.strip()
	parastr = "".join(para)
	begins, ends = map(list, zip(*[y for x in offsets for y in x]))

	best_dist = 1e200
	best_indices = None

	if span == parastr:
		return parastr, (0, len(parastr)), 0

	for m in re.finditer(re.escape(span), parastr):
		begin_offset, end_offset = m.span()

		fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < end_offset)
		fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > begin_offset)

		if d1 + d2 < best_dist:
			best_dist = d1 + d2
			best_indices = (fixed_begin, fixed_end)
			if best_dist == 0:
				break

	if best_indices is None:
		return None, None, best_dist
	return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist


def word_tokenize(sent):
	return tokenizer.tokenize(sent)


def convert_idx(text, tokens):
	current = 0
	spans = []
	for token in tokens:
		current = text.find(token, current)
		if current < 0:
			raise Exception()
		spans.append((current, current + len(token)))
		current += len(token)
	return spans


def _process_para(para, sp_set):
	text_context, context_tokens = '', []
	offsets = []
	flat_offsets = []
	start_end_facts = []  # (start_token_id, end_token_id, is_sup_fact=True/False)
	sent2title_ids = []

	def _process(sent, is_sup_fact, N_chars, is_title=False):
		nonlocal text_context, context_tokens, offsets, start_end_facts, flat_offsets

		sent_tokens = word_tokenize(sent)
		if is_title:
			sent_tokens = sent_tokens + [':']
		sent = ' '.join(sent_tokens)
		sent_spans = convert_idx(sent, sent_tokens)

		sent_spans = [[N_chars + e[0], N_chars + e[1]] for e in sent_spans]
		N_tokens, my_N_tokens = len(context_tokens), len(sent_tokens)

		text_context += sent
		context_tokens.extend(sent_tokens)
		start_end_facts.append((N_tokens, N_tokens + my_N_tokens, is_sup_fact))
		offsets.append(sent_spans)
		flat_offsets.extend(sent_spans)

	cur_title, cur_para = para[0], para[1]
	_process(cur_title, False, len(text_context), is_title=True)
	sent2title_ids.append((cur_title, -1))
	for sent_id, sent in enumerate(cur_para):
		is_sup_fact = (cur_title, sent_id) in sp_set
		_process(sent, is_sup_fact, len(text_context))
		sent2title_ids.append((cur_title, sent_id))

	return text_context, context_tokens, offsets, flat_offsets, start_end_facts, sent2title_ids


def overlap_span(start, end, y1, y2):
	if start <= y1 < end or start <= y2 < end:
		return True
	return y1 <= start and y2 >= end


def fix_start_end_facts(start_end_facts, y1, y2):
	assert y1[0] == y2[0]
	fixed_start_end_facts = []
	for para_id, start_end_fact in enumerate(start_end_facts):
		fixed_start_end_fact = []
		if para_id == y1[0]:
			for start, end, is_sp in start_end_fact:
				if overlap_span(start, end, y1[1], y2[1]):
					fixed_start_end_fact.append((start, end, True))
				else:
					fixed_start_end_fact.append((start, end, is_sp))
		else:
			fixed_start_end_fact = start_end_fact
		fixed_start_end_facts.append(fixed_start_end_fact)
	return fixed_start_end_facts


def _process_article(article):
	paragraphs = article['context']
	# some articles in the fullwiki dev/test sets have zero paragraphs
	if len(paragraphs) == 0:
		return None

	text_context, context_tokens = [], []
	offsets = []
	flat_offsets = []
	start_end_facts = []  # (start_token_id, end_token_id, is_sup_fact=True/False)
	sent2title_ids = []

	if 'supporting_facts' in article:
		sp_set = set(list(map(tuple, article['supporting_facts'])))
	else:
		sp_set = set()

	for para in paragraphs:
		text_context_para, context_tokens_para, offsets_para, flat_offsets_para, start_end_facts_para, sent2title_ids_para = _process_para(
			para, sp_set)
		text_context.append(text_context_para)
		context_tokens.append(context_tokens_para)
		offsets.append(offsets_para)
		flat_offsets.append(flat_offsets_para)
		start_end_facts.append(start_end_facts_para)
		sent2title_ids.append(sent2title_ids_para)

	if 'answer' in article:
		answer = ' '.join(word_tokenize(article['answer'].strip()))
		if answer.lower() == 'yes':
			best_indices = ((-1, -1), (-1, -1))
		elif answer.lower() == 'no':
			best_indices = ((-1, -2), (-1, -2))
		else:
			if answer not in ''.join(text_context):
				# in the fullwiki setting, the answer might not have been retrieved
				# use (0, 1) so that we can proceed
				best_indices = ((-1, 0), (-1, 1))
			else:
				triples = [(para_id, *fix_span(text_context_para, offsets_para, answer)) for
						   para_id, (text_context_para, offsets_para) in enumerate(zip(text_context, offsets))]
				triples.sort(key=lambda e: e[3])
				best_para, _, best_indices, _ = triples[0]
				assert best_indices is not None
				answer_span = []
				for idx, span in enumerate(flat_offsets[best_para]):
					if not (best_indices[1] <= span[0] or best_indices[0] >= span[1]):
						answer_span.append((best_para, idx))
				best_indices = (answer_span[0], answer_span[-1])
				start_end_facts = fix_start_end_facts(start_end_facts, best_indices[0], best_indices[1])
	else:
		# some random stuff
		answer = 'random'
		best_indices = ((-1, -2), (-1, -2))

	ques_tokens = word_tokenize(article['question'])

	example = {'context_tokens': context_tokens, 'ques_tokens': ques_tokens,
			   'y1s': [best_indices[0]], 'y2s': [best_indices[1]], 'id': article['_id'],
			   'start_end_facts': start_end_facts}
	eval_example = {'context': text_context, 'question': (article['question']), 'spans': flat_offsets,
					'answer': [answer], 'id': article['_id'], 'sent2title_ids': sent2title_ids}
	return example, eval_example


def process_file(filename):
	data = json.load(open(filename, 'r'))

	eval_examples = {}

	outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_article)(article) for article in data)
	# outputs = [_process_article(article, config) for article in data]
	outputs = [output for output in outputs if output is not None]
	examples = [e[0] for e in outputs]
	for _, e in outputs:
		if e is not None:
			eval_examples[e['id']] = e

	random.shuffle(examples)
	print("{} questions in total".format(len(examples)))

	return examples, eval_examples


def convert_tokens_to_ids(tokens):
	return [tokenizer.vocab[token] if token in tokenizer.vocab else tokenizer.vocab[UNK] for token in tokens]


def build_features(examples, data_type, out_file):
	def filter_func(example):
		return 3 + len(example["context_tokens"]) + len(example["ques_tokens"]) > MAX_SEQ_LEN

	print("Processing {} examples...".format(data_type))
	datapoints = []
	total = 0
	total_ = 0
	for example in tqdm(examples):
		total_ += 1
		if filter_func(example):
			continue
		total += 1
		context_idxs = [torch.tensor(convert_tokens_to_ids(para)) for para in example['context_tokens']]
		ques_idxs = torch.tensor(convert_tokens_to_ids(example['ques_tokens']))

		start, end = example["y1s"][-1], example["y2s"][-1]
		y1, y2 = start, end

		datapoints.append({
			'context_tokens': example['context_tokens'],
			'ques_tokens': example['ques_tokens'],
			'context_idxs': context_idxs,
			'ques_idxs': ques_idxs,
			'y1': y1,
			'y2': y2,
			'id': example['id'],
			'start_end_facts': example['start_end_facts']})
	print("Build {} / {} instances of features in total".format(total, total_))
	# pickle.dump(datapoints, open(out_file, 'wb'), protocol=-1)
	torch.save(datapoints, out_file)


def save(filename, obj, message=None):
	if message is not None:
		print("Saving {}...".format(message))
	with open(filename, "w") as fh:
		json.dump(obj, fh)


def prepro(config):
	random.seed(13)

	examples, eval_examples = process_file(config.data_file)

	if config.data_split == 'train':
		record_file = config.train_record_file
		eval_file = config.train_eval_file
	elif config.data_split == 'dev':
		record_file = config.dev_record_file
		eval_file = config.dev_eval_file
	elif config.data_split == 'test':
		record_file = config.test_record_file
		eval_file = config.test_eval_file

	build_features(examples, config.data_split, record_file)
	save(eval_file, eval_examples, message='{} eval'.format(config.data_split))
