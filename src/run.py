import math
import os
import shutil
import time

import ujson as json
from torch import optim, nn

from model.hop_model import HOPModel
from utils.iterator import *

nll = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)

	print('Experiment dir : {}'.format(path))
	if scripts_to_save is not None:
		if not os.path.exists(os.path.join(path, 'scripts')):
			os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)


def unpack(data):
	full_batch = data[FULL_BATCH_KEY]
	context_ques_idxs = data[CONTEXT_QUES_IDXS_KEY]
	compact_context_ques_idxs = data[COMPACT_CONTEXT_QUES_IDXS_KEY]
	context_ques_masks = data[CONTEXT_QUES_MASKS_KEY]
	compact_context_ques_masks = data[COMPACT_CONTEXT_QUES_MASKS_KEY]
	context_ques_segments = data[CONTEXT_QUES_SEGMENTS_KEY]
	compact_context_ques_segments = data[COMPACT_CONTEXT_QUES_SEGMENTS_KEY]
	compact_answer_masks = data[COMPACT_ANSWER_MASKS_KEY]
	q_type = data[Q_TYPE_KEY]
	is_support = data[IS_SUPPORT_KEY]
	compact_is_support = data[COMPACT_IS_SUPPORT_KEY]
	has_support = data[HAS_SP_KEY]
	all_mapping = data[ALL_MAPPING_KEY]
	compact_all_mapping = data[COMPACT_ALL_MAPPING_KEY]
	y1 = data[Y1_KEY]
	y2 = data[Y2_KEY]
	compact_y1 = data[COMPACT_Y1_KEY]
	compact_y2 = data[COMPACT_Y2_KEY]
	compact_to_orig_mapping = data[COMPACT_TO_ORIG_MAPPING_KEY]
	return full_batch, \
		   context_ques_idxs, compact_context_ques_idxs, \
		   context_ques_masks, compact_context_ques_masks, \
		   context_ques_segments, compact_context_ques_segments, \
		   compact_answer_masks, \
		   is_support, compact_is_support, has_support, \
		   all_mapping, compact_all_mapping, \
		   compact_y1, compact_y2, q_type, y1, y2, \
		   compact_to_orig_mapping


def build_iterator(config, bucket, batch_size, shuffle):
	return DataIterator(bucket, batch_size, shuffle, debug=config.debug)


def train(config):
	with open(config.dev_eval_file, "r") as fh:
		dev_eval_file = json.load(fh)

	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed_all(config.seed)

	config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
	create_exp_dir(config.save)

	def logging(s, print_=True, log_=True):
		if print_:
			print(s)
		if log_:
			with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
				f_log.write(s + '\n')

	logging('Config')
	for k, v in config.__dict__.items():
		logging('    - {} : {}'.format(k, v))

	logging("Building model...")
	train_datapoints = get_datapoitns(config.train_record_file)
	dev_datapoints = get_datapoitns(config.dev_record_file)

	model = HOPModel(config)

	logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
	ori_model = model if config.debug else model.cuda()
	model = nn.DataParallel(ori_model)

	if config.debug:
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
	else:
		optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, model.module.base.parameters())},
								{'params': filter(lambda p: p.requires_grad, model.module.bert.parameters()),
								 'lr': config.bert_lr}], lr=config.init_lr)
	total_loss = 0
	global_step = 0
	best_dev_F1 = None
	stop_train = False
	start_time = time.time()
	eval_start_time = time.time()
	model.train()
	optimizer.zero_grad()

	for epoch in range(config.epoch):
		for data in build_iterator(config, train_datapoints, config.batch_size, not config.debug):
			full_batch, \
			context_ques_idxs, compact_context_ques_idxs, \
			context_ques_masks, compact_context_ques_masks, \
			context_ques_segments, compact_context_ques_segments, \
			compact_answer_masks, \
			is_support, compact_is_support, has_support, \
			all_mapping, compact_all_mapping, \
			compact_y1, compact_y2, q_type, y1, y2, \
			compact_to_orig_mapping \
				= unpack(data)

			# has_support_logits, is_support_logits \
			# 	= model(context_ques_idxs, context_ques_masks, context_ques_segments, None, all_mapping, task='locate')

			start_logits, end_logits, type_logits, compact_is_support_logits \
				= model(compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
						compact_answer_masks, compact_all_mapping, task='reason')
			#
			# loss_sp = (nll(is_support_logits.view(-1, 2), is_support.view(-1)) +
			# 		   nll(has_support_logits, has_support.view(-1)) +
			# 		   nll(compact_is_support_logits.view(-1, 2), compact_is_support.view(-1))) / \
			# 		  config.aggregate_step
			loss_ans = (nll(start_logits, compact_y1) +
						nll(end_logits, compact_y2) +
						nll(type_logits, q_type)) / config.aggregate_step
			# loss = config.sp_lambda * loss_sp + config.ans_lambda * loss_ans
			loss = config.ans_lambda * loss_ans
			loss.backward()
			total_loss += loss.item()

			if (global_step + 1) & config.aggregate_step == 0:
				optimizer.step()
				optimizer.zero_grad()
			global_step += 1

			if global_step % config.period == 0:
				cur_loss = total_loss * config.aggregate_step / config.period
				elapsed = time.time() - start_time
				logging('| epoch {:3d} | step {:6d} | ms/batch {:5.2f} | train loss {:8.3f}'.format(
					epoch, global_step, elapsed * 1000 / config.period, cur_loss))
				total_loss = 0
				start_time = time.time()

			if global_step % config.checkpoint == 0:
				model.eval()
				metrics = evaluate_batch(
					build_iterator(config, dev_datapoints, math.ceil(config.batch_size), False),
					model, 1 if config.debug else 0, dev_eval_file, config)
				model.train()

				logging('-' * 89)
				logging(
					'| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f} | SP_Precision {:.4f} | SP_Recall {:.4f} | SP_F1 {:.4f}'.format(
						global_step // config.checkpoint, epoch, time.time() - eval_start_time,
						metrics['loss'], metrics['exact_match'], metrics['f1'], metrics['sp_precision'],
						metrics['sp_recall'], metrics['sp_f1']))
				logging('-' * 89)

				eval_start_time = time.time()

				dev_F1 = metrics['f1']
				if best_dev_F1 is None or dev_F1 > best_dev_F1:
					best_dev_F1 = dev_F1
					torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
				if stop_train: break
	logging('best_dev_F1 {}'.format(best_dev_F1))


def build_reasoner_input(full_batch, has_support_logits, cuda, ground_truth=False):
	if ground_truth:
		para_idxs = [[i for i, has_sp_fact in enumerate(data[HAS_SP_KEY]) if has_sp_fact] for data in full_batch]
	else:
		para_idxs = []
		sorted_para_idxs = list(np.argsort(-has_support_logits, axis=1))
		for data_i, data in enumerate(full_batch):
			para_idx = []
			cur_ctx_ques_size = 2 + len(data[QUES_IDXS_KEY])
			for para_i in sorted_para_idxs[data_i]:
				para_idx.append(para_i)
				cur_ctx_ques_size += len(data[CONTEXT_IDXS_KEY][para_i])
				if cur_ctx_ques_size >= BERT_LIMIT:
					break
			para_idxs.append(para_idx)
	return filter_para(full_batch, para_idxs, cuda)


def map_compact_to_orig_y(yp, compact_to_orig_mapping):
	bsz = yp.shape[0]
	return compact_to_orig_mapping[np.arange(bsz), yp]


@torch.no_grad()
def evaluate_batch(data_source, model, max_batches, eval_file, config):
	answer_dict = {}
	sp_pred, sp_true = [], []
	total_loss, step_cnt = 0, 0
	iter = data_source
	for step, data in enumerate(iter):
		if step >= max_batches and max_batches > 0: break

		full_batch, \
		context_ques_idxs, compact_context_ques_idxs, \
		context_ques_masks, compact_context_ques_masks, \
		context_ques_segments, compact_context_ques_segments, \
		compact_answer_masks, \
		is_support, compact_is_support, has_support, \
		all_mapping, compact_all_mapping, \
		compact_y1, compact_y2, q_type, y1, y2, \
		compact_to_orig_mapping \
			= unpack(data)

		has_support_logits, is_support_logits \
			= model(context_ques_idxs, context_ques_masks, context_ques_segments, None, all_mapping, task='locate')

		start_logits, end_logits, type_logits, compact_is_support_logits \
			= model(compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
					compact_answer_masks, compact_all_mapping, task='reason')

		# loss_sp = (nll(is_support_logits.view(-1, 2), is_support.view(-1)) +
		# 		   nll(has_support_logits, has_support.view(-1)) +
		# 		   nll(compact_is_support_logits.view(-1, 2), compact_is_support.view(-1)))
		loss_ans = (nll(start_logits, compact_y1) +
					nll(end_logits, compact_y2) +
					nll(type_logits, q_type))
		# loss = config.sp_lambda * loss_sp + config.ans_lambda * loss_ans
		loss = config.ans_lambda * loss_ans
		total_loss += loss.item()
		compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, \
		compact_answer_masks, compact_all_mapping, compact_to_orig_mapping \
			= build_reasoner_input(full_batch, has_support_logits.data.cpu().numpy(), not config.debug,
								   ground_truth=config.sp_lambda <= 0.0)
		start_logits, end_logits, type_logits, _, yp1, yp2 \
			= model(compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
					compact_answer_masks, compact_all_mapping, task='reason', return_yp=True)
		yp1 = map_compact_to_orig_y(yp1.data.cpu().numpy(), compact_to_orig_mapping)
		yp2 = map_compact_to_orig_y(yp2.data.cpu().numpy(), compact_to_orig_mapping)

		answer_dict_ = convert_tokens(eval_file, data['ids'], yp1, yp2, np.argmax(type_logits.data.cpu().numpy(), 1))
		answer_dict.update(answer_dict_)

		predict_support_np \
			= np.rint(torch.sigmoid(is_support_logits[:, :, :, 1]).data.cpu().numpy()).astype(int).flatten()
		is_support_np = is_support.data.cpu().numpy().flatten()
		assert len(is_support_np) == len(predict_support_np)
		for sp_t, sp_p in zip(is_support_np, predict_support_np):
			if sp_t == IGNORE_INDEX:
				continue
			sp_true.append(sp_t)
			sp_pred.append(sp_p)

		step_cnt += 1
	loss = total_loss / step_cnt
	metrics = evaluate(eval_file, answer_dict)
	metrics['loss'] = loss
	sp_precision, sp_recall, sp_f1 = evaluate_sp(sp_true, sp_pred)
	metrics['sp_precision'] = sp_precision
	metrics['sp_recall'] = sp_recall
	metrics['sp_f1'] = sp_f1
	return metrics
