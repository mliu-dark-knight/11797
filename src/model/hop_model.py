import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertConfig, BertLayer, BertPooler, gelu
from torch.autograd import Variable

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
		self.intermediate_hidden = nn.Linear(self.bert_hidden, config.hidden_size)
		bert_config = BertConfig(-1, hidden_size=config.hidden_size, num_hidden_layers=1,
								 num_attention_heads=config.num_attention_heads,
								 intermediate_size=config.intermediate_size)
		if config.locate_global:
			self.encoder_has_support = BertLayer(bert_config)
			self.pooler_has_support = BertPooler(bert_config)
			self.linear_has_support = nn.Linear(config.hidden_size, 1)
		else:
			self.linear_has_support = nn.Linear(self.bert_hidden, 1)
		self.encoder_is_support = BertLayer(bert_config)
		self.linear_is_support = nn.Linear(config.hidden_size, 1)
		self.is_support_to_span = nn.Linear(2 * config.hidden_size, config.hidden_size)
		self.encoder_span = BertLayer(bert_config)
		self.pooler_type = BertPooler(bert_config)
		self.linear_span = nn.Linear(config.hidden_size, 2)
		self.linear_type = nn.Linear(config.hidden_size, 3)

	def get_output_mask(self, outer):
		S = outer.size(1)
		np_mask = np.tril(np.triu(np.ones((S, S)), 0), ANS_LIMIT - 1)
		cache_mask = outer.new_tensor(np_mask)
		return Variable(cache_mask, requires_grad=False)

	def forward(self, context_ques_idxs, context_ques_masks, context_ques_segments, answer_masks, all_mapping,
				task='reason', return_yp=False):
		assert task in ['locate', 'reason']
		bsz, para_cnt, token_cnt \
			= context_ques_idxs.size(0), context_ques_idxs.size(1), context_ques_idxs.size(2)
		if self.config.debug:
			bert_output, pooled_output \
				= next(self.parameters()).new_tensor(np.random.rand(bsz * para_cnt, token_cnt, self.bert_hidden)), \
				  next(self.parameters()).new_tensor(np.random.rand(bsz * para_cnt, self.bert_hidden))
		else:
			bert_output, pooled_output \
				= self.bert(context_ques_idxs.view(bsz * para_cnt, token_cnt),
							context_ques_segments.view(bsz * para_cnt, token_cnt),
							context_ques_masks.view(bsz * para_cnt, token_cnt),
							output_all_encoded_layers=False)

		extended_attention_mask = context_ques_masks.view(bsz * para_cnt, token_cnt).unsqueeze(1).unsqueeze(2)
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		if task == 'locate':
			if self.config.locate_global:
				bert_output = self.intermediate_hidden(bert_output)
				bert_output = gelu(bert_output)
				bert_output = self.encoder_has_support(bert_output, extended_attention_mask)
				pooled_output = self.pooler_has_support(bert_output)
			one_logits = self.linear_has_support(pooled_output)
			zero_logits = torch.zeros_like(one_logits)
			has_support_logits = torch.cat((zero_logits, one_logits), dim=1)
			return has_support_logits.view(bsz, para_cnt, 2)

		bert_output = self.intermediate_hidden(bert_output)
		bert_output = gelu(bert_output)

		answer_masks = answer_masks.squeeze(dim=1)
		all_mapping = all_mapping.squeeze(dim=1)

		is_support_output = self.encoder_is_support(bert_output, extended_attention_mask)
		is_support_input = torch.div(torch.bmm(all_mapping, is_support_output),
		                             torch.sum(all_mapping, dim=2, keepdim=True) + SMALL_FLOAT)
		one_logits = self.linear_is_support(is_support_input)
		zero_logits = torch.zeros_like(one_logits)
		is_support_logits = torch.cat((zero_logits, one_logits), dim=2).unsqueeze(dim=1)

		is_support_to_span = torch.bmm(all_mapping.permute(0, 2, 1), is_support_input)
		is_support_to_span = gelu(self.is_support_to_span(torch.cat((bert_output, is_support_to_span), dim=2)))
		span_input = self.encoder_span(is_support_to_span, extended_attention_mask)
		type_input = self.pooler_type(span_input)
		type_logits = self.linear_type(type_input)
		span_logits = self.linear_span(span_input)
		span_logits -= (1. - answer_masks.unsqueeze(dim=2)) * BIG_INT
		start_logits, end_logits = span_logits.split(1, dim=2)
		start_logits, end_logits = start_logits.squeeze(dim=2), end_logits.squeeze(dim=2)
		if not return_yp:
			return start_logits, end_logits, type_logits, is_support_logits

		outer = start_logits.unsqueeze(dim=2) + end_logits.unsqueeze(dim=1)
		outer_mask = self.get_output_mask(outer).unsqueeze(dim=0)
		outer -= BIG_INT * (1. - outer_mask)
		yp1 = outer.max(dim=2)[0].max(dim=1)[1]
		yp2 = outer.max(dim=1)[0].max(dim=1)[1]
		return start_logits, end_logits, type_logits, is_support_logits, yp1, yp2
