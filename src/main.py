import argparse
import os

from prepro import prepro
from run import train, test

parser = argparse.ArgumentParser()

data_dir = '../data'

train_eval = os.path.join(data_dir, "train_eval.json")
dev_eval = os.path.join(data_dir, "dev_eval.json")
test_eval = os.path.join(data_dir, "test_eval.json")
train_record_file = os.path.join(data_dir, 'train_record.pkl')
dev_record_file = os.path.join(data_dir, 'dev_record.pkl')
test_record_file = os.path.join(data_dir, 'test_record.pkl')

parser.add_argument('--debug', default=True, action='store_true')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--fp16', default=False, action='store_true')
# parser.add_argument('--data_file', type=str, default='../data/hotpot_dev_distractor_v1.json')
# parser.add_argument('--save', type=str, default='../experiment/HOTPOT-20190424-234152')
parser.add_argument('--data_file', type=str)
parser.add_argument('--save', type=str, default='../experiment/HOTPOT')

parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)

parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--aggregate_step', type=int, default=8)
parser.add_argument('--epoch', type=int, default=4)
parser.add_argument('--checkpoint', type=int, default=8000)
parser.add_argument('--period', type=int, default=800)
parser.add_argument('--init_lr', type=float, default=1e-4)
parser.add_argument('--bert_lr', type=float, default=2e-5)
parser.add_argument('--compact_para_cnt', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--intermediate_size', type=int, default=256)
parser.add_argument('--locate_global', default=False, action='store_true', help='Add global layer for locator')
parser.add_argument('--reason_top', default=False, action='store_true', help='Add top BERT layer for reasoner')
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--ans_lambda', type=float, default=1.0)
parser.add_argument('--is_sp_lambda', type=float, default=1.0)
parser.add_argument('--has_sp_lambda', type=float, default=1.0)

parser.add_argument('--data_split', type=str, default='train')
parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--sp_threshold', type=float, default=0.3)
# parser.add_argument('--prediction_file', type=str, default='../data/dev_distractor_pred.json')
parser.add_argument('--prediction_file', type=str)

config = parser.parse_args()


def _concat(filename):
	if config.fullwiki:
		path, name = os.path.split(filename)
		return os.path.join(path, 'fullwiki.{}'.format(name))
	return filename


config.dev_record_file = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file = _concat(config.dev_eval_file)
config.test_eval_file = _concat(config.test_eval_file)
if config.debug:
	config.batch_size = 4
	config.aggregate_step = 2
	config.checkpoint = 1
	config.period = 1

if __name__ == '__main__':
	if config.mode == 'train':
		train(config)
	elif config.mode == 'prepro':
		prepro(config)
	elif config.mode == 'test':
		test(config)
