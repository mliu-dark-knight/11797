import torch

if __name__ == '__main__':
	dev_record = torch.load('../data/dev_record.pkl')
	dev_wiki_record = torch.load('../data/fullwiki.dev_record.pkl')
	test_record = torch.load('../data/fullwiki.test_record.pkl')
	dev_record = dev_record[: int(len(dev_record) / 100)]
	dev_wiki_record = dev_wiki_record[: int(len(dev_wiki_record) / 100)]
	test_record = test_record[: int(len(test_record) / 100)]
	torch.save(dev_record, '../data/dev_record.pkl')
	torch.save(dev_wiki_record, '../data/fullwiki.dev_record.pkl')
	torch.save(test_record, '../data/fullwiki.test_record.pkl')
	pass
