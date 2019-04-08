import math
import os
import shutil
import time

import ujson as json
from torch import optim, nn
from torch.autograd import Variable

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
	context_ques_masks = data[CONTEXT_QUES_MASKS_KEY]
	context_ques_segments = data[CONTEXT_QUES_SEGMENTS_KEY]
	answer_masks = data[ANSWER_MASKS_KEY]
	ques_size = data[QUES_SIZE_KEY]
	q_type = data[Q_TYPE_KEY]
	is_support = data[IS_SUPPORT_KEY]
	all_mapping = data[ALL_MAPPING_KEY]
	y1 = data[Y1_KEY]
	y2 = data[Y2_KEY]
	y1_flat = data[Y1_FLAT_KEY]
	y2_flat = data[Y2_FLAT_KEY]
	return full_batch, context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, ques_size, q_type, is_support, all_mapping, y1, y2, y1_flat, y2_flat


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

	lr = config.init_lr
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
	total_loss = 0
	global_step = 0
	best_dev_F1 = None
	stop_train = False
	start_time = time.time()
	eval_start_time = time.time()
	model.train()

	for epoch in range(config.epoch):
		for data in build_iterator(config, train_datapoints, config.batch_size, not config.debug):
			_, context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, ques_size, q_type, is_support, all_mapping, y1, y2, y1_flat, y2_flat \
				= unpack(data)

			start_logits, end_logits, type_logits, support_logits \
				= model(context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, all_mapping)
			loss = nll(start_logits, y1_flat) + nll(end_logits, y2_flat) + nll(type_logits, q_type) + \
				   config.sp_lambda * nll(support_logits.view(-1, 2), is_support.view(-1))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			global_step += 1

			if global_step % config.period == 0:
				cur_loss = total_loss / config.period
				elapsed = time.time() - start_time
				logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(
					epoch, global_step, lr, elapsed * 1000 / config.period, cur_loss))
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
						metrics['loss'], metrics['exact_match'], metrics['f1'], metrics['sp_precision'], metrics['sp_recall'], metrics['sp_f1']))
				logging('-' * 89)

				eval_start_time = time.time()

				dev_F1 = metrics['f1']
				if best_dev_F1 is None or dev_F1 > best_dev_F1:
					best_dev_F1 = dev_F1
					torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
				if stop_train: break
	logging('best_dev_F1 {}'.format(best_dev_F1))


def unflatten_y(y, max_ctx_ques_size, ques_size):
	y1 = (y / max_ctx_ques_size).astype(int)
	y2 = y % max_ctx_ques_size
	y2 = y2 - ques_size - 2
	return np.stack((y1, y2), axis=1)


@torch.no_grad()
def evaluate_batch(data_source, model, max_batches, eval_file, config):
	answer_dict = {}
	sp_pred, sp_true = [], []
	total_loss, step_cnt = 0, 0
	iter = data_source
	for step, data in enumerate(iter):
		if step >= max_batches and max_batches > 0: break

		_, context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, ques_size, q_type, is_support, all_mapping, y1, y2, y1_flat, y2_flat \
			= unpack(data)

		start_logits, end_logits, type_logits, support_logits, yp1, yp2 \
			= model(context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, all_mapping,
					return_yp=True)
		loss = nll(start_logits, y1_flat) + nll(end_logits, y2_flat) + nll(type_logits, q_type) + \
			   config.sp_lambda * nll(support_logits.view(-1, 2), is_support.view(-1))

		max_ctx_ques_size = context_ques_idxs.size(2)
		yp1 = unflatten_y(yp1.data.cpu().numpy(), max_ctx_ques_size, ques_size)
		yp2 = unflatten_y(yp2.data.cpu().numpy(), max_ctx_ques_size, ques_size)

		answer_dict_ = convert_tokens(eval_file, data['ids'], yp1, yp2, np.argmax(type_logits.data.cpu().numpy(), 1))
		answer_dict.update(answer_dict_)

		is_support_np = is_support.data.cpu().numpy().flatten()
		predict_support_np = np.rint(torch.sigmoid(support_logits[:, :, 1]).data.cpu().numpy().flatten()).astype(int)
		for sp_t, sp_p in zip(is_support_np, predict_support_np):
			if sp_t == IGNORE_INDEX:
				continue
			sp_true.append(sp_t)
			sp_pred.append(sp_p)

		total_loss += loss.item()
		step_cnt += 1
	loss = total_loss / step_cnt
	metrics = evaluate(eval_file, answer_dict)
	metrics['loss'] = loss
	sp_precision, sp_recall, sp_f1 = evaluate_sp(sp_true, sp_pred)
	metrics['sp_precision'] = sp_precision
	metrics['sp_recall'] = sp_recall
	metrics['sp_f1'] = sp_f1
	return metrics
