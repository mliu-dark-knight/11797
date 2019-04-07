from pytorch_pretrained_bert import BertModel

from model.common import *
from utils.constants import *


class HOPModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.hidden = config.hidden

		self.bert = BertModel.from_pretrained('bert-base-uncased')

		self.linear_span = nn.Linear(config.hidden, 2)
		self.linear_type = nn.Linear(config.hidden * 2, 3)

		self.cache_S = 0

	def forward(self, context_ques_idxs, context_ques_masks, context_ques_segments, return_yp=False):
		bsz, para_cnt = context_ques_idxs.size(0), context_ques_idxs.size(1)
		bert_output = self.bert(
			context_ques_idxs.view(bsz * para_cnt, context_ques_idxs.size(2), context_ques_idxs.size(3)),
			context_ques_segments.view(bsz * para_cnt, context_ques_idxs.size(2), context_ques_idxs.size(3)),
			context_ques_masks.view(bsz * para_cnt, context_ques_idxs.size(2), context_ques_idxs.size(3)),
			output_all_encoded_layers=False)
		type_input = bert_output.view(bsz, para_cnt, context_ques_idxs.size(2), context_ques_idxs.size(3))
		type_input = torch.max(type_input[:, :, 0, :], dim=1)
		type_logits = self.linear_type(type_input)
		span_logits = self.linear_span(bert_output)
		span_logits -= (1 - context_ques_masks) * BIG_INT
		start_logits, end_logits = span_logits.split(1, dim=1)
		return start_logits, end_logits, type_logits
