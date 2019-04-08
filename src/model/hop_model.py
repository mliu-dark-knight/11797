import numpy as np
from pytorch_pretrained_bert import BertModel

from model.common import *
from utils.constants import *


class HOPModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		if self.config.debug:
			self.bert_hidden = BERT_HIDDEN
		else:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			self.bert_hidden = self.bert.config.hidden_size
		self.linear_span = nn.Linear(self.bert_hidden, 2)
		self.linear_type = nn.Linear(self.bert_hidden, 3)

	def get_output_mask(self, outer):
		S = outer.size(1)
		np_mask = np.tril(np.triu(np.ones((S, S)), 0), ANS_LIMIT - 1)
		cache_mask = outer.new_tensor(np_mask)
		return Variable(cache_mask, requires_grad=False)

	def forward(self, context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, return_yp=False):
		bsz, para_cnt, para_limit = context_ques_idxs.size(0), context_ques_idxs.size(1), context_ques_idxs.size(2)
		if self.config.debug:
			bert_output, pooled_output \
				= next(self.parameters()).new_tensor(np.random.rand(bsz * para_cnt, para_limit, self.bert_hidden)), \
				  next(self.parameters()).new_tensor(np.random.rand(bsz * para_cnt, self.bert_hidden))
		else:
			bert_output, pooled_output \
				= self.bert(context_ques_idxs.view(bsz * para_cnt, para_limit),
							context_ques_segments.view(bsz * para_cnt, para_limit),
							context_ques_masks.view(bsz * para_cnt, para_limit),
							output_all_encoded_layers=False)
		type_input = torch.max(pooled_output.view(bsz, para_cnt, self.bert_hidden), dim=1)[0]
		type_logits = self.linear_type(type_input)
		span_input = bert_output.view(bsz, para_cnt * para_limit, self.bert_hidden)
		span_logits = self.linear_span(span_input)
		span_logits -= (1. - answer_masks.view(bsz, para_cnt * para_limit).unsqueeze(dim=2)) * BIG_INT
		start_logits, end_logits = span_logits.split(1, dim=2)
		start_logits, end_logits = start_logits.squeeze(dim=2), end_logits.squeeze(dim=2)
		if not return_yp:
			return start_logits, end_logits, type_logits

		outer = start_logits.unsqueeze(dim=2) + end_logits.unsqueeze(dim=1)
		outer_mask = self.get_output_mask(outer).unsqueeze(dim=0)
		outer -= BIG_INT * (1. - outer_mask)
		yp1 = outer.max(dim=2)[0].max(dim=1)[1]
		yp2 = outer.max(dim=1)[0].max(dim=1)[1]
		return start_logits, end_logits, type_logits, yp1, yp2
