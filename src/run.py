import os
import random
import shutil
import time
import ujson as json

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from tqdm import tqdm

from model.hop_model import HOPModel
from model.sp_model import SPModel
from utils.eval import convert_tokens, evaluate, evaluate_sp, get_buckets, IGNORE_INDEX
from utils.iterator import DataIterator


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


def model_output(model, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs,
                 context_lens, start_mapping, end_mapping, all_mapping, baseline=False, return_yp=False):
	if baseline:
		if return_yp:
			logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs,
			                                                                context_char_idxs, ques_char_idxs,
			                                                                context_lens,
			                                                                start_mapping, end_mapping, all_mapping,
			                                                                return_yp=return_yp)
		else:
			logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs,
			                                                      context_char_idxs, ques_char_idxs, context_lens,
			                                                      start_mapping, end_mapping, all_mapping,
			                                                      return_yp=return_yp)
	else:
		predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs,
		                        context_lens, start_mapping, end_mapping, stage='locate', return_yp=return_yp)
		if return_yp:
			logit1, logit2, predict_type, yp1, yp2 = model(context_idxs_r, ques_idxs, context_char_idxs_r,
			                                               ques_char_idxs, None, None, None,
			                                               stage='reason', return_yp=return_yp)
		else:
			logit1, logit2, predict_type = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
			                                     None, None, None, stage='reason', return_yp=return_yp)

	if return_yp:
		return logit1, logit2, predict_support, predict_type, yp1, yp2
	return logit1, logit2, predict_support, predict_type


def unpack(data):
	context_idxs = data['context_idxs']
	context_idxs_r = data['context_idxs_r']
	ques_idxs = data['ques_idxs']
	context_char_idxs = data['context_char_idxs']
	context_char_idxs_r = data['context_char_idxs_r']
	ques_char_idxs = data['ques_char_idxs']
	context_lens = data['context_lens']
	y1 = data['y1']
	y1_r = data['y1_r']
	y2 = data['y2']
	y2_r = data['y2_r']
	y_offsets = data['y_offsets']
	y_offsets_r = data['y_offsets_r']
	q_type = Variable(data['q_type'])
	is_support = Variable(data['is_support'])
	start_mapping = data['start_mapping']
	end_mapping = data['end_mapping']
	all_mapping = data['all_mapping']
	return context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, context_lens, \
	       y1, y1_r, y2, y2_r, y_offsets, y_offsets_r, q_type, is_support, start_mapping, end_mapping, all_mapping


nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduce=False, ignore_index=IGNORE_INDEX)


def train(config):
	if config.debug:
		word_mat = np.random.rand(395261, 300)
	else:
		with open(config.word_emb_file, "r") as fh:
			word_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.char_emb_file, "r") as fh:
		char_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.dev_eval_file, "r") as fh:
		dev_eval_file = json.load(fh)
	with open(config.idx2word_file, 'r') as fh:
		idx2word_dict = json.load(fh)

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
	train_buckets = get_buckets(config.train_record_file)
	dev_buckets = get_buckets(config.dev_record_file)

	def build_train_iterator():
		return DataIterator(train_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit,
		                    True, config.sent_limit, len(word_mat), len(char_mat), debug=config.debug, p=config.p)

	def build_dev_iterator():
		return DataIterator(dev_buckets, config.batch_size, config.para_limit, config.ques_limit, config.char_limit,
		                    False, config.sent_limit, len(word_mat), len(char_mat), debug=config.debug, p=config.p)

	if config.baseline:
		model = SPModel(config, word_mat, char_mat)
	else:
		model = HOPModel(config, word_mat, char_mat)

	logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
	ori_model = model if config.debug else model.cuda()
	model = nn.DataParallel(ori_model)

	lr = config.init_lr
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
	cur_patience = 0
	total_loss = 0
	global_step = 0
	best_dev_F1 = None
	stop_train = False
	start_time = time.time()
	eval_start_time = time.time()
	model.train()

	for epoch in range(10000):
		for data in build_train_iterator():
			context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, context_lens, \
			y1, y1_r, y2, y2_r, y_offsets, y_offsets_r, q_type, is_support, start_mapping, end_mapping, all_mapping = unpack(
				data)

			logit1, logit2, predict_support, predict_type = model_output(
				model, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs,
				context_lens, start_mapping, end_mapping, all_mapping, baseline=config.baseline, return_yp=False)

			if not config.baseline:
				y1 = y1_r
				y2 = y2_r
			loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) +
			          nll_sum(logit2, y2)) / context_idxs.size(0)
			loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
			loss = loss_1 + config.sp_lambda * loss_2

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
				metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file, config)
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
					cur_patience = 0
				else:
					cur_patience += 1
					if cur_patience >= config.patience:
						lr /= 2.0
						for param_group in optimizer.param_groups:
							param_group['lr'] = lr
						if lr < config.init_lr * 1e-2:
							stop_train = True
							break
						cur_patience = 0
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

		context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, context_lens, \
		y1, y1_r, y2, y2_r, y_offsets, y_offsets_r, q_type, is_support, start_mapping, end_mapping, all_mapping \
			= unpack(data)

		logit1, logit2, predict_support, predict_type, yp1, yp2 = model_output(
			model, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs,
			context_lens, start_mapping, end_mapping, all_mapping, baseline=config.baseline, return_yp=True)

		if not config.baseline:
			y_offsets = y_offsets_r
			y1 = y1_r
			y2 = y2_r
		loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + \
		       config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))

		answer_dict_ = convert_tokens(eval_file, data['ids'],
		                              yp1.data.cpu().numpy() + y_offsets,
		                              yp2.data.cpu().numpy() + y_offsets,
		                              np.argmax(predict_type.data.cpu().numpy(), 1))
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


@torch.no_grad()
def predict(data_source, model, eval_file, config, prediction_file):
	answer_dict = {}
	sp_dict = {}
	sp_th = config.sp_threshold
	for step, data in enumerate(tqdm(data_source)):
		context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, context_lens, \
		y1, y1_r, y2, y2_r, y_offsets, y_offsets_r, q_type, is_support, start_mapping, end_mapping, all_mapping \
			= unpack(data)

		logit1, logit2, predict_support, predict_type, yp1, yp2 = model_output(
			model, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs,
			context_lens, start_mapping, end_mapping, all_mapping, baseline=config.baseline, return_yp=True)

		if not config.baseline: y_offsets = y_offsets_r
		answer_dict_ = convert_tokens(eval_file, data['ids'],
		                              yp1.data.cpu().numpy() + y_offsets,
		                              yp2.data.cpu().numpy() + y_offsets,
		                              np.argmax(predict_type.data.cpu().numpy(), 1))
		answer_dict.update(answer_dict_)

		predict_support_np = torch.sigmoid(predict_support[:, :, 1]).data.cpu().numpy()
		for i in range(predict_support_np.shape[0]):
			cur_sp_pred = []
			cur_id = data['ids'][i]
			for j in range(predict_support_np.shape[1]):
				if j >= len(eval_file[cur_id]['sent2title_ids']): break
				if predict_support_np[i, j] > sp_th:
					cur_sp_pred.append(eval_file[cur_id]['sent2title_ids'][j])
			sp_dict.update({cur_id: cur_sp_pred})

	prediction = {'answer': answer_dict, 'sp': sp_dict}
	with open(prediction_file, 'w') as f:
		json.dump(prediction, f)


def test(config):
	with open(config.word_emb_file, "r") as fh:
		word_mat = np.array(json.load(fh), dtype=np.float32)
	with open(config.char_emb_file, "r") as fh:
		char_mat = np.array(json.load(fh), dtype=np.float32)
	if config.data_split == 'dev':
		with open(config.dev_eval_file, "r") as fh:
			dev_eval_file = json.load(fh)
	else:
		with open(config.test_eval_file, 'r') as fh:
			dev_eval_file = json.load(fh)
	with open(config.idx2word_file, 'r') as fh:
		idx2word_dict = json.load(fh)

	random.seed(config.seed)
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed_all(config.seed)

	def logging(s, print_=True, log_=True):
		if print_:
			print(s)
		if log_:
			with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
				f_log.write(s + '\n')

	if config.data_split == 'dev':
		dev_buckets = get_buckets(config.dev_record_file)
		para_limit = config.para_limit
		ques_limit = config.ques_limit
	elif config.data_split == 'test':
		para_limit = None
		ques_limit = None
		dev_buckets = get_buckets(config.test_record_file)

	def build_dev_iterator():
		return DataIterator(dev_buckets, config.batch_size, para_limit,
		                    ques_limit, config.char_limit, False, config.sent_limit, len(word_mat), len(char_mat),
		                    debug=config.debug, p=0.0)

	if config.baseline:
		model = SPModel(config, word_mat, char_mat)
	else:
		model = HOPModel(config, word_mat, char_mat)
	ori_model = model if config.debug else model.cuda()
	ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
	model = nn.DataParallel(ori_model)

	model.eval()
	predict(build_dev_iterator(), model, dev_eval_file, config, config.prediction_file)
