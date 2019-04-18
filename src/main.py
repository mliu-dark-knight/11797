import argparse
import os

from prepro import prepro
from run import train, test

parser = argparse.ArgumentParser()

data_dir = '../data'

glove_word_file = os.path.join(data_dir, "glove.840B.300d.txt")

word_emb_file = os.path.join(data_dir, "word_emb.json")
char_emb_file = os.path.join(data_dir, "char_emb.json")
train_eval =  os.path.join(data_dir, "train_eval.json")
dev_eval = os.path.join(data_dir, "dev_eval.json")
test_eval = os.path.join(data_dir, "test_eval.json")
word2idx_file = os.path.join(data_dir, "word2idx.json")
char2idx_file = os.path.join(data_dir, "char2idx.json")
idx2word_file = os.path.join(data_dir, 'idx2word.json')
idx2char_file = os.path.join(data_dir, 'idx2char.json')
train_record_file = os.path.join(data_dir, 'train_record.pkl')
dev_record_file = os.path.join(data_dir, 'dev_record.pkl')
test_record_file = os.path.join(data_dir, 'test_record.pkl')


parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--cpu', default=False, action='store_true')
parser.add_argument('--p', type=float, default=0.8)
parser.add_argument('--use_gt', default=False, action='store_true')
parser.add_argument('--baseline', default=False, action='store_true')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_file', type=str)
parser.add_argument('--glove_word_file', type=str, default=glove_word_file)
parser.add_argument('--save', type=str, default='../experiment/HOTPOT')

parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
parser.add_argument('--char_emb_file', type=str, default=char_emb_file)
parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
parser.add_argument('--char2idx_file', type=str, default=char2idx_file)
parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
parser.add_argument('--idx2char_file', type=str, default=idx2char_file)

parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)

parser.add_argument('--glove_char_size', type=int, default=94)
parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
parser.add_argument('--glove_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=8)

parser.add_argument('--para_limit', type=int, default=2250)
parser.add_argument('--ques_limit', type=int, default=80)
parser.add_argument('--char_limit', type=int, default=16)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epoch', type=int, default=4)
parser.add_argument('--checkpoint', type=int, default=1000)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=1e-3)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--hidden', type=int, default=80)
parser.add_argument('--char_hidden', type=int, default=100)
parser.add_argument('--reason_step', type=int, default=1)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--sp_lambda', type=float, default=1.0)

parser.add_argument('--data_split', type=str, default='train')
parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--prediction_file', type=str)
parser.add_argument('--sp_threshold', type=float, default=0.3)

config = parser.parse_args()

def _concat(filename):
    if config.fullwiki:
        path, name = os.path.split(filename)
        return os.path.join(path, 'fullwiki.{}'.format(name))
    return filename
# config.train_record_file = _concat(config.train_record_file)
config.dev_record_file = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
# config.train_eval_file = _concat(config.train_eval_file)
config.dev_eval_file = _concat(config.dev_eval_file)
config.test_eval_file = _concat(config.test_eval_file)
if config.debug:
    config.batch_size = 4
    config.checkpoint = 1
    config.period = 1
    config.train_record_file = config.dev_record_file
    config.train_eval_file = config.dev_eval_file

if __name__ == '__main__':
    if config.mode == 'train':
        train(config)
    elif config.mode == 'prepro':
        prepro(config)
    elif config.mode == 'test':
        assert not config.baseline
        test(config)
