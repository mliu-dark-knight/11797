import math
import os
import shutil
import time

import numpy as np
import ujson as json
from torch import optim, nn
from torch.autograd import Variable
from tqdm import tqdm

from model.hop_model import HOPModel
from utils.iterator import *

nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)


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



def model_output(config, model, full_batch, context_idxs, context_idxs_r, ques_idxs, context_char_idxs,
                 context_char_idxs_r, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping,
                 orig_idxs, orig_idxs_r, return_yp=False):
	predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs,
	                        context_lens, start_mapping, end_mapping, stage='locate', return_yp=return_yp)
	if return_yp:
		if config.use_gt:
			logit1, logit2, predict_type, yp1, yp2 \
				= model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
				        None, None, None, stage='reason', return_yp=True)
		else:
			logit1, logit2, _, = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
			                           None, None, None, stage='reason', return_yp=False)
			batch_p = torch.sigmoid(predict_support[:, :, 1]).data.cpu() < config.sp_threshold
			para_limit = context_idxs.size()[1]
			char_limit = config.char_limit
			cpu = config.cpu
			cur_batch = sample_sent(full_batch, para_limit, char_limit, batch_p=batch_p, force_drop=True)

			context_idxs_r, context_char_idxs_r, _, _, _, _, _, orig_idxs_r \
				= build_ctx_tensor(cur_batch, char_limit, not cpu)
			_, _, predict_type, yp1, yp2 = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
			                                     None, None, None, stage='reason', return_yp=True)
		return logit1, logit2, predict_support, predict_type, \
		       orig_idxs[np.arange(len(yp1)), orig_idxs_r[np.arange(len(yp1)), yp1.data.cpu().numpy()]], \
		       orig_idxs[np.arange(len(yp2)), orig_idxs_r[np.arange(len(yp2)), yp2.data.cpu().numpy()]]

	logit1, logit2, predict_type = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
	                                     None, None, None, stage='reason', return_yp=False)

	return logit1, logit2, predict_support, predict_type


def unpack(data):
	full_batch = data[FULL_BATCH_KEY]
	context_ques_idxs = data[CONTEXT_QUES_IDXS_KEY]
	context_ques_masks = data[CONTEXT_QUES_MASKS_KEY]
	context_ques_segments = data[CONTEXT_QUES_SEGMENTS_KEY]
	q_type = Variable(data[Q_TYPE_KEY])
	y1 = data[Y1_KEY]
	y2 = data[Y2_KEY]
	y1_flat = data[Y1_FLAT_KEY]
	y2_flat = data[Y2_FLAT_KEY]
	return full_batch, context_ques_idxs, context_ques_masks, context_ques_segments, q_type, y1, y2, y1_flat, y2_flat


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
	ori_model = model if config.cpu else model.cuda()
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
			_, context_ques_idxs, context_ques_masks, context_ques_segments, q_type, y1, y2, y1_flat, y2_flat = unpack(data)

			start_logits, end_logits, type_logits = model(context_ques_idxs, context_ques_masks, context_ques_segments)
			loss = nll_average(start_logits, y1_flat) + nll_average(end_logits, y2_flat) + nll_average(type_logits, q_type)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			global_step += 1

			if global_step % config.period == 0:
				cur_loss = total_loss / config.period
				elapsed = time.time() - start_time
				logging('| epoch {:3d} | step {:6d} | lr {:05.5f} | ms/batch {:5.2f} | train loss {:8.3f}'.format(
					epoch, global_step, lr, elapsed * 1000 / config.period, cur_loss))
				start_time = time.time()

			if global_step % config.checkpoint == 0:
				model.eval()
				metrics = evaluate_batch(
					build_iterator(config, dev_datapoints, math.ceil(config.batch_size), False),
					model, 1 if config.debug else 0, dev_eval_file, config)
				model.train()

				logging('-' * 89)
				logging(
					'| eval {:6d} in epoch {:3d} | time: {:5.2f}s | dev loss {:8.3f} | EM {:.4f} | F1 {:.4f} | SP_F1 {:.4f}'.format(
						global_step // config.checkpoint, epoch, time.time() - eval_start_time,
						metrics['loss'], metrics['exact_match'], metrics['f1'], metrics['sp_f1']))
				logging('-' * 89)

				eval_start_time = time.time()

				dev_F1 = metrics['f1']
				if best_dev_F1 is None or dev_F1 > best_dev_F1:
					best_dev_F1 = dev_F1
					torch.save(ori_model.state_dict(), os.path.join(config.save, 'model.pt'))
				if stop_train: break
	logging('best_dev_F1 {}'.format(best_dev_F1))


@torch.no_grad()
def evaluate_batch(data_source, model, max_batches, eval_file, config):
	answer_dict = {}
	sp_pred, sp_true = [], []
	total_loss, step_cnt = 0, 0
	iter = data_source
	for step, data in enumerate(iter):
		if step >= max_batches and max_batches > 0: break

		_, context_ques_idxs, context_ques_masks, context_ques_segments, q_type, y1, y2, y1_flat, y2_flat = unpack(data)


		answer_dict_ = convert_tokens(eval_file, data['ids'], yp1, yp2, np.argmax(predict_type.data.cpu().numpy(), 1))
		answer_dict.update(answer_dict_)

		is_support_np = is_support.data.cpu().numpy().flatten()
		predict_support_np = np.rint(torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy().flatten()).astype(int)
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
	metrics['sp_f1'] = evaluate_sp(sp_true, sp_pred)
	return metrics
