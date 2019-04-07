import torch
import ujson

if __name__ == '__main__':
	dev_eval_json = '../data/test_eval.json'
	dev_record_pkl = '../data/test_record.pkl'
	with open(dev_eval_json, "r") as fh:
		dev_eval_file = ujson.load(fh)
	datapoints = torch.load(dev_record_pkl)
	new_dev_eval_file = dict()
	new_datapoints = datapoints[: int(len(datapoints) / 100)]
	for datapoint in new_datapoints:
		id = datapoint['id']
		new_dev_eval_file[id] = dev_eval_file[id]
	with open(dev_eval_json, 'w') as f:
		ujson.dump(new_dev_eval_file, f)
	torch.save(new_datapoints, dev_record_pkl)

