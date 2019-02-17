import torch
import ujson as json


def subsample_eval(orig_eval, records):
	new_eval = {}
	for record in records:
		new_eval[record['id']] = orig_eval[record['id']]
	return new_eval

if __name__ == '__main__':
	dev_record = torch.load('../data/dev_record.pkl')
	dev_wiki_record = torch.load('../data/fullwiki.dev_record.pkl')
	test_record = torch.load('../data/fullwiki.test_record.pkl')
	with open('../data/dev_eval.json', 'r') as fh:
		dev_eval_file = json.load(fh)
	with open('../data/fullwiki.dev_eval.json', 'r') as fh:
		dev_wiki_eval_file = json.load(fh)
	with open('../data/fullwiki.test_eval.json', 'r') as fh:
		test_eval_file = json.load(fh)
	dev_eval_file = subsample_eval(dev_eval_file, dev_record)
	dev_wiki_eval_file = subsample_eval(dev_wiki_eval_file, dev_wiki_record)
	test_eval_file = subsample_eval(test_eval_file, test_record)
	with open('../data/dev_eval.json', 'w') as fh:
		json.dump(dev_eval_file, fh)
	with open('../data/fullwiki.dev_eval.json', 'w') as fh:
		json.dump(dev_wiki_eval_file, fh)
	with open('../data/fullwiki.test_eval.json', 'w') as fh:
		json.dump(test_eval_file, fh)
	pass
