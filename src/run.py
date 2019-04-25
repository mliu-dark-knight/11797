import os
import shutil
import time

import ujson as json
from torch import optim, nn
from tqdm import tqdm

try:
	from apex.parallel import DistributedDataParallel as DDP
	from apex import amp
except ImportError:
	# raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
	pass


from model.hop_model import HOPModel
from utils.iterator import *

nll = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def set_random_seed(config):
	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed_all(config.seed)


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
		   compact_all_mapping, \
		   compact_y1, compact_y2, q_type, y1, y2, \
		   compact_to_orig_mapping


def build_iterator(config, bucket, shuffle):
	return DataIterator(bucket, config.batch_size, shuffle, config.compact_para_cnt, debug=config.debug)


def get_model_logits(model, context_ques_idxs, context_ques_masks, context_ques_segments,
					 compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
					 compact_all_mapping, compact_answer_masks):
	has_support_logits \
		= model(context_ques_idxs, context_ques_masks, context_ques_segments, None, None, task='locate')
	start_logits, end_logits, type_logits, compact_is_support_logits \
		= model(compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
				compact_answer_masks, compact_all_mapping, task='reason')
	return has_support_logits, compact_is_support_logits, start_logits, end_logits, type_logits


def get_model_loss(has_support, compact_is_support, compact_y1, compact_y2, q_type,
				   has_support_logits, compact_is_support_logits, start_logits, end_logits, type_logits):
	loss_has_sp = nll(has_support_logits.view(-1, 2), has_support.view(-1))
	loss_is_sp = nll(compact_is_support_logits.view(-1, 2), compact_is_support.view(-1))
	loss_ans = nll(start_logits, compact_y1) + nll(end_logits, compact_y2) + nll(type_logits, q_type)
	return loss_has_sp, loss_is_sp, loss_ans


def logging(config, s, print_=True, log_=True):
	if print_:
		print(s)
	if log_:
		with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
			f_log.write(s + '\n')


def train(config):
	with open(config.dev_eval_file, "r") as fh:
		dev_eval_file = json.load(fh)

	set_random_seed(config)

	config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
	create_exp_dir(config.save)

	logging(config, 'Config')
	for k, v in config.__dict__.items():
		logging(config, '    - {} : {}'.format(k, v))

	logging(config, "Building model...")
	train_datapoints = get_datapoints(config.train_record_file)
	dev_datapoints = get_datapoints(config.dev_record_file)

	model = HOPModel(config)
	if not config.debug: model = model.cuda()

	logging(config, 'nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))

	if config.debug:
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
	else:
		optimizer = optim.Adam([{'params': [param for name, param in model.named_parameters()
											if 'bert' not in name and param.requires_grad]},
								{'params': filter(lambda p: p.requires_grad, model.bert.parameters()),
								 'lr': config.bert_lr}], lr=config.init_lr)

	if not config.debug and config.fp16:
		model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
		model = DDP(model, delay_allreduce=True)
	else:
		model = nn.DataParallel(model)

	total_loss = 0
	global_step = 0
	best_dev_F1 = None
	stop_train = False
	start_time = time.time()
	eval_start_time = time.time()
	model.train()
	optimizer.zero_grad()

	for epoch in range(config.epoch):
		for data in build_iterator(config, train_datapoints, not config.debug):
			full_batch, \
			context_ques_idxs, compact_context_ques_idxs, \
			context_ques_masks, compact_context_ques_masks, \
			context_ques_segments, compact_context_ques_segments, \
			compact_answer_masks, \
			is_support, compact_is_support, has_support, \
			compact_all_mapping, \
			compact_y1, compact_y2, q_type, y1, y2, \
			compact_to_orig_mapping \
				= unpack(data)

			has_support_logits, compact_is_support_logits, start_logits, end_logits, type_logits \
				= get_model_logits(
				model, context_ques_idxs, context_ques_masks, context_ques_segments,
				compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
				compact_all_mapping, compact_answer_masks)

			loss_has_sp, loss_is_sp, loss_ans \
				= get_model_loss(has_support, compact_is_support, compact_y1, compact_y2, q_type,
								 has_support_logits, compact_is_support_logits, start_logits, end_logits, type_logits)
			loss = (config.has_sp_lambda * loss_has_sp +
					config.is_sp_lambda * loss_is_sp +
					config.ans_lambda * loss_ans) / config.aggregate_step

			loss.backward()
			total_loss += loss.item()

			if (global_step + 1) % config.aggregate_step == 0:
				optimizer.step()
				optimizer.zero_grad()
			global_step += 1

			if global_step % config.period == 0:
				cur_loss = total_loss * config.aggregate_step / config.period
				elapsed = time.time() - start_time
				logging(config, '| epoch {:3d} | step {:6d} | ms/batch {:5.2f} | train loss {:8.3f}'.format(
					epoch, global_step, elapsed * 1000 / config.period, cur_loss))
				total_loss = 0
				start_time = time.time()

			if global_step % config.checkpoint == 0:
				model.eval()
				metrics = evaluate_batch(
					build_iterator(config, dev_datapoints, False), model, 2 if config.debug else 0, dev_eval_file,
					config)
				model.train()

				logging(config, '-' * 89)
				logging(config,
						'| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f} | HAS_SP_Precision {:.4f} | HAS_SP_Recall {:.4f} | HAS_SP_F1 {:.4f} | IS_SP_F1 {:.4f}'.format(
							global_step // config.checkpoint, epoch, time.time() - eval_start_time,
							metrics['loss'], metrics['exact_match'], metrics['f1'],
							metrics['has_sp_precision'], metrics['has_sp_recall'], metrics['has_sp_f1'],
							metrics['is_sp_f1']))
				logging(config, '-' * 89)

				eval_start_time = time.time()

				dev_F1 = metrics['f1']
				if best_dev_F1 is None or dev_F1 > best_dev_F1:
					best_dev_F1 = dev_F1
					torch.save(model.module.state_dict(), os.path.join(config.save, 'model.pt'))
				if stop_train: break
	logging(config, 'best_dev_F1 {}'.format(best_dev_F1))


def select_reasoner_para(config, full_batch, has_support, ground_truth=False):
	if ground_truth:
		para_idxs = get_mixed_para_idxs(full_batch, config.compact_para_cnt)
	else:
		para_idxs = []
		sorted_para_idxs = list(np.argsort(-has_support, axis=1))
		for data_i, data in enumerate(full_batch):
			para_idx = []
			cur_ctx_ques_size = 3 + len(data[QUES_IDXS_KEY])
			for para_i in sorted_para_idxs[data_i]:
				if len(para_idx) >= config.compact_para_cnt:
					break
				# some data points may have less than 10 paragraphs
				if para_i < len(data[HAS_SP_KEY]) and \
						cur_ctx_ques_size + len(data[CONTEXT_IDXS_KEY][para_i]) <= BERT_LIMIT:
					para_idx.append(para_i)
					cur_ctx_ques_size += len(data[CONTEXT_IDXS_KEY][para_i])
			para_idxs.append(para_idx)
	return para_idxs


def map_compact_to_orig_y(yp, compact_to_orig_mapping):
	bsz = yp.shape[0]
	return compact_to_orig_mapping['token'][np.arange(bsz), yp]


def map_compact_to_orig_sp(sp, compact_to_orig_mapping):
	bsz, sent_cnt = sp.shape
	return compact_to_orig_mapping['sent'][np.expand_dims(np.arange(bsz), axis=1),
										   np.expand_dims(np.arange(sent_cnt), axis=0)]


@torch.no_grad()
def evaluate_batch(data_source, model, max_batches, eval_file, config):
	answer_dict = {}
	is_sp_pred, is_sp_true = [], []
	has_sp_pred, has_sp_true = [], []
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
		compact_all_mapping, \
		compact_y1, compact_y2, q_type, y1, y2, \
		compact_to_orig_mapping \
			= unpack(data)

		has_support_logits, compact_is_support_logits, start_logits, end_logits, type_logits \
			= get_model_logits(
			model, context_ques_idxs, context_ques_masks, context_ques_segments,
			compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
			compact_all_mapping, compact_answer_masks)

		loss_has_sp, loss_is_sp, loss_ans \
			= get_model_loss(has_support, compact_is_support, compact_y1, compact_y2, q_type,
							 has_support_logits, compact_is_support_logits, start_logits, end_logits, type_logits)
		loss = (config.has_sp_lambda * loss_has_sp +
				config.is_sp_lambda * loss_is_sp +
				config.ans_lambda * loss_ans)
		total_loss += loss.item()

		para_idxs \
			= select_reasoner_para(config, full_batch, has_support_logits[:, :, 1].data.cpu().numpy(),
								   ground_truth=config.has_sp_lambda <= 0.0)
		compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, \
		compact_answer_masks, compact_all_mapping, compact_to_orig_mapping \
			= build_compact_tensor_no_support(full_batch, para_idxs, not config.debug)
		start_logits, end_logits, type_logits, compact_is_support_logits, yp1, yp2 \
			= model(compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
					compact_answer_masks, compact_all_mapping, task='reason', return_yp=True)
		yp1 = map_compact_to_orig_y(yp1.data.cpu().numpy(), compact_to_orig_mapping)
		yp2 = map_compact_to_orig_y(yp2.data.cpu().numpy(), compact_to_orig_mapping)

		answer_dict_ = convert_tokens(eval_file, data['ids'], yp1, yp2, np.argmax(type_logits.data.cpu().numpy(), 1))
		answer_dict.update(answer_dict_)

		is_support_np = is_support.data.cpu().numpy()
		compact_predict_support_np = np.rint(torch.sigmoid(
			compact_is_support_logits[:, :, :, 1]).squeeze(dim=1).data.cpu().numpy()).astype(int)
		predict_support_np = get_pred_sp_np(is_support_np, compact_predict_support_np, compact_to_orig_mapping)
		is_support_np = is_support_np.flatten()
		predict_support_np = predict_support_np.flatten()

		for sp_t, sp_p in zip(is_support_np, predict_support_np):
			if sp_t == IGNORE_INDEX:
				continue
			is_sp_true.append(sp_t)
			is_sp_pred.append(sp_p)

		has_support_np = has_support.data.cpu().numpy()
		for data_i, para_idx in enumerate(para_idxs):
			for para_i, has_sp in enumerate(has_support_np[data_i]):
				if has_sp == IGNORE_INDEX:
					continue
				has_sp_true.append(has_sp)
				if para_i in para_idx:
					has_sp_pred.append(1)
				else:
					has_sp_pred.append(0)

		step_cnt += 1
	loss = total_loss / step_cnt
	metrics = evaluate(eval_file, answer_dict)
	metrics['loss'] = loss
	_, _, sp_f1 = evaluate_sp(is_sp_true, is_sp_pred)
	has_sp_precision, has_sp_recall, has_sp_f1 = evaluate_sp(has_sp_true, has_sp_pred)
	metrics['is_sp_f1'] = sp_f1
	metrics['has_sp_precision'] = has_sp_precision
	metrics['has_sp_recall'] = has_sp_recall
	metrics['has_sp_f1'] = has_sp_f1
	return metrics


def get_pred_sp_np(is_support_np, compact_predict_support_np, compact_to_orig_mapping):
	predict_support_np = np.zeros_like(is_support_np)
	pred_sp_indices = map_compact_to_orig_sp(compact_predict_support_np, compact_to_orig_mapping)
	filter = np.logical_and(pred_sp_indices[:, :, 0] != INVALID_INDEX, pred_sp_indices[:, :, 1] != INVALID_INDEX)
	predict_support_np[
		np.repeat(np.arange(pred_sp_indices.shape[0])[:, None], pred_sp_indices.shape[1], axis=1)[filter],
		pred_sp_indices[:, :, 0][filter],
		pred_sp_indices[:, :, 1][filter]] = compact_predict_support_np[filter]
	return predict_support_np


@torch.no_grad()
def predict(data_source, model, max_batches, eval_file, config):
	answer_dict = {}
	sp_dict = {}
	sp_th = config.sp_threshold
	for step, data in enumerate(tqdm(data_source)):
		if step >= max_batches and max_batches > 0: break

		full_batch, \
		context_ques_idxs, compact_context_ques_idxs, \
		context_ques_masks, compact_context_ques_masks, \
		context_ques_segments, compact_context_ques_segments, \
		compact_answer_masks, \
		is_support, compact_is_support, has_support, \
		compact_all_mapping, \
		compact_y1, compact_y2, q_type, y1, y2, \
		compact_to_orig_mapping \
			= unpack(data)

		has_support_logits \
			= model(context_ques_idxs, context_ques_masks, context_ques_segments, None, None, task='locate')

		para_idxs \
			= select_reasoner_para(config, full_batch, has_support_logits[:, :, 1].data.cpu().numpy(),
								   ground_truth=config.has_sp_lambda <= 0.0)
		compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments, \
		compact_answer_masks, compact_all_mapping, compact_to_orig_mapping \
			= build_compact_tensor_no_support(full_batch, para_idxs, not config.debug)
		start_logits, end_logits, type_logits, compact_is_support_logits, yp1, yp2 \
			= model(compact_context_ques_idxs, compact_context_ques_masks, compact_context_ques_segments,
					compact_answer_masks, compact_all_mapping, task='reason', return_yp=True)
		yp1 = map_compact_to_orig_y(yp1.data.cpu().numpy(), compact_to_orig_mapping)
		yp2 = map_compact_to_orig_y(yp2.data.cpu().numpy(), compact_to_orig_mapping)

		answer_dict_ = convert_tokens(eval_file, data['ids'], yp1, yp2, np.argmax(type_logits.data.cpu().numpy(), 1))
		answer_dict.update(answer_dict_)

		is_support_np = is_support.data.cpu().numpy()
		compact_predict_support_np = np.rint(torch.sigmoid(
			compact_is_support_logits[:, :, :, 1]).squeeze(dim=1).data.cpu().numpy()).astype(int)
		predict_support_np = get_pred_sp_np(is_support_np, compact_predict_support_np, compact_to_orig_mapping)

		for i in range(predict_support_np.shape[0]):
			cur_sp_pred = []
			cur_id = data['ids'][i]
			for j in range(predict_support_np.shape[1]):
				if j >= len(eval_file[cur_id]['sent2title_ids']): break
				for k in range(predict_support_np.shape[2]):
					if k >= len(eval_file[cur_id]['sent2title_ids'][j]): break
					if predict_support_np[i, j, k] > sp_th:
						cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j][k])
			sp_dict.update({cur_id: cur_sp_pred})

	prediction = {'answer': answer_dict, 'sp': sp_dict}
	with open(config.prediction_file, 'w') as f:
		json.dump(prediction, f)


def test(config):
	with open(eval('config.' + config.data_split + '_eval_file'), "r") as fh:
		dev_eval_file = json.load(fh)

	set_random_seed(config)
	dev_datapoints = get_datapoints(eval('config.' + config.data_split + '_record_file'))

	model = HOPModel(config)
	if not config.debug: model = model.cuda()
	model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
	model = nn.DataParallel(model)
	model.eval()

	if config.prediction_file is not None:
		predict(build_iterator(config, dev_datapoints, False), model, 2 if config.debug else 0, dev_eval_file, config)
	else:
		eval_start_time = time.time()
		metrics = evaluate_batch(build_iterator(config, dev_datapoints, False), model, 2 if config.debug else 0,
								 dev_eval_file, config)
		logging(config, '-' * 89)
		logging(config,
				'| time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f} | HAS_SP_Precision {:.4f} | HAS_SP_Recall {:.4f} | HAS_SP_F1 {:.4f} | IS_SP_F1 {:.4f}'.format(
					time.time() - eval_start_time,
					metrics['loss'], metrics['exact_match'], metrics['f1'],
					metrics['has_sp_precision'], metrics['has_sp_recall'], metrics['has_sp_f1'],
					metrics['is_sp_f1']))
		logging(config, '-' * 89)
