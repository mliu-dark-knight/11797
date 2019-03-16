import math
import os
import shutil
import time
import ujson as json

from torch import optim, nn
from torch.autograd import Variable
from tqdm import tqdm

from model.hop_model import HOPModel
from utils.iterator import *

nll_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=IGNORE_INDEX)
nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
nll_all = nn.CrossEntropyLoss(reduce=False, ignore_index=IGNORE_INDEX)


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
                 context_char_idxs_r, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, y_offsets,
                 return_yp=False):
	predict_support = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs,
	                        context_lens, start_mapping, end_mapping, stage='locate', return_yp=return_yp)
	if return_yp:
		logit1, logit2, _, _, _ = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
		                                None, None, None, stage='reason', return_yp=return_yp)
		batch_p = torch.sigmoid(predict_support[:, :, 1]).data.cpu() < config.sp_threshold
		para_limit = context_idxs.size()[1]
		char_limit = config.char_limit
		debug = config.debug
		cur_batch = sample_sent(full_batch, para_limit, char_limit, batch_p=batch_p)

		context_idxs_r, context_char_idxs_r, _, _, _, _, _ \
			= build_ctx_tensor(cur_batch, char_limit, not debug)
		_, _, _, y_offsets_r = build_ans_tensor(cur_batch, not debug)
		y_offsets_r += y_offsets

		_, _, predict_type, yp1, yp2 = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
		                                     None, None, None, stage='reason', return_yp=return_yp)
		return logit1, logit2, predict_support, predict_type, \
		       yp1.data.cpu().numpy() + y_offsets_r, yp2.data.cpu().numpy() + y_offsets_r

	logit1, logit2, predict_type = model(context_idxs_r, ques_idxs, context_char_idxs_r, ques_char_idxs,
	                                     None, None, None, stage='reason', return_yp=return_yp)

	return logit1, logit2, predict_support, predict_type


def unpack(data):
	full_batch = data[FULL_BATCH_KEY]
	context_idxs = data[CONTEXT_IDXS_KEY]
	context_idxs_r = data[CONTEXT_IDXS_R_KEY]
	ques_idxs = data[QUES_IDXS_KEY]
	context_char_idxs = data[CONTEXT_CHAR_IDXS_KEY]
	context_char_idxs_r = data[CONTEXT_CHAR_IDXS_R_KEY]
	ques_char_idxs = data[QUES_CHAR_IDXS_KEY]
	context_lens = data[CONTEXT_LENS_KEY]
	y1_r = data[Y1_R_KEY]
	y2_r = data[Y2_R_KEY]
	y_offsets = data[Y_OFFSETS_KEY]
	q_type = Variable(data[Q_TYPE_KEY])
	is_support = Variable(data[IS_SUPPORT_KEY])
	start_mapping = data[START_MAPPING_KEY]
	end_mapping = data[END_MAPPING_KEY]
	all_mapping = data[ALL_MAPPING_KEY]
	return full_batch, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, \
	       context_lens, y1_r, y2_r, y_offsets, q_type, is_support, start_mapping, end_mapping, all_mapping


def build_iterator(config, buckets, batch_size, shuffle, num_word, num_char, p):
	return DataIterator(buckets, batch_size, config.para_limit, config.ques_limit, config.char_limit,
	                    shuffle, num_word, num_char, debug=config.debug, p=p)


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

	model = HOPModel(config, word_mat, char_mat)

	logging('nparams {}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
	ori_model = model if config.debug else model.cuda()
	model = nn.DataParallel(ori_model)

	lr = config.init_lr
	optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
	total_loss = 0
	global_step = 0
	best_dev_F1 = None
	stop_train = False
	start_time = time.time()
	eval_start_time = time.time()
	model.train()

	for epoch in range(config.epoch):
		for data in build_iterator(config, train_buckets, config.batch_size, not config.debug, len(word_mat),
		                           len(char_mat), config.p):
			_, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, \
			context_lens, y1_r, y2_r, _, q_type, is_support, start_mapping, end_mapping, all_mapping = unpack(data)

			logit1, logit2, predict_support, predict_type = model_output(
				config, model, None, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r,
				ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, None, return_yp=False)

			loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1_r) +
			          nll_sum(logit2, y2_r)) / context_idxs.size(0)
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
				metrics = evaluate_batch(
					build_iterator(config, dev_buckets, math.ceil(config.batch_size / 2), False, len(word_mat),
					               len(char_mat), config.p),
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

		full_batch, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, \
		context_lens, y1_r, y2_r, y_offsets, q_type, is_support, start_mapping, end_mapping, all_mapping = unpack(
			data)

		logit1, logit2, predict_support, predict_type, yp1, yp2 = model_output(
			config, model, full_batch, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r,
			ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, y_offsets, return_yp=True)

		loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1_r) + nll_sum(logit2, y2_r)) / context_idxs.size(0) + \
		       config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))

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


@torch.no_grad()
def predict(data_source, model, eval_file, config, prediction_file):
	answer_dict = {}
	sp_dict = {}
	sp_th = config.sp_threshold
	for step, data in enumerate(tqdm(data_source)):
		full_batch, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r, ques_char_idxs, \
		context_lens, y1_r, y2_r, y_offsets, q_type, is_support, start_mapping, end_mapping, all_mapping = unpack(data)

		logit1, logit2, predict_support, predict_type, yp1, yp2 = model_output(
			config, model, full_batch, context_idxs, context_idxs_r, ques_idxs, context_char_idxs, context_char_idxs_r,
			ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, y_offsets, return_yp=True)

		answer_dict_ = convert_tokens(eval_file, data['ids'], yp1, yp2, np.argmax(predict_type.data.cpu().numpy(), 1))
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

	model = HOPModel(config, word_mat, char_mat)
	ori_model = model if config.debug else model.cuda()
	ori_model.load_state_dict(torch.load(os.path.join(config.save, 'model.pt')))
	model = nn.DataParallel(ori_model)

	model.eval()
	predict(
		build_iterator(config, math.ceil(config.batch_size / 2), dev_buckets, False, len(word_mat), len(char_mat), 0.0),
		model, dev_eval_file, config, config.prediction_file)
