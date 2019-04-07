import numpy as np

from model.common import *
from pytorch_pretrained_bert import BertModel


class Memory(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.rnn = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1 - config.keep_prob, False)
		self.self_att = BiAttention(config.hidden * 2, 1 - config.keep_prob)
		self.linear = nn.Sequential(
			nn.Linear(config.hidden * 8, config.hidden),
			nn.ReLU()
		)

	def forward(self, input, context_lens, context_mask):
		output_t = self.rnn(input, context_lens)
		output_t = self.self_att(output_t, output_t, context_mask)
		output_t = self.linear(output_t)
		return input + output_t



class HOPModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.hidden = config.hidden

		self.bert = BertModel.from_pretrained('bert-base-uncased')

		self.memory = Memory(config)

		self.rnn_sp = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1 - config.keep_prob, False)
		self.linear_sp = nn.Linear(config.hidden * 2, 1)

		self.rnn_start = EncoderRNN(config.hidden * 2, config.hidden, 1, False, True, 1 - config.keep_prob, False)
		self.linear_start = nn.Linear(config.hidden * 2, 1)

		self.rnn_end = EncoderRNN(config.hidden * 3, config.hidden, 1, False, True, 1 - config.keep_prob, False)
		self.linear_end = nn.Linear(config.hidden * 2, 1)

		self.rnn_type = EncoderRNN(config.hidden * 3, config.hidden, 1, False, True, 1 - config.keep_prob, False)
		self.linear_type = nn.Linear(config.hidden * 2, 4)

		self.cache_S = 0

	def get_output_mask(self, outer):
		S = outer.size(1)
		if S <= self.cache_S:
			return Variable(self.cache_mask[:S, :S], requires_grad=False)
		self.cache_S = S
		np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
		self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
		return Variable(self.cache_mask, requires_grad=False)

	def forward(self, context_ques_idxs, segment_idxs, start_mapping, end_mapping, stage='locate', return_yp=False):
		para_size, bsz = context_ques_idxs.size(1), context_ques_idxs.size(0)
		bert_output = self.bert(context_ques_idxs, segment_idxs)

		context_mask = (context_idxs > 0).float()

		context_word = self.word_emb(context_idxs)

		context_output = torch.cat((context_word, context_ch), dim=2)

		context_output = self.rnn(context_output, context_lens)

		output = self.qc_att(context_output, ques_output, ques_mask)
		output = self.linear_1(output)

		output_memory = self.memory(output, context_lens, context_mask)

		if stage == 'locate':
			sp_output = self.rnn_sp(output, context_lens)

			start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:, :, self.hidden:])
			end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:, :, :self.hidden])
			sp_output = torch.cat((start_output, end_output), dim=-1)
			sp_output_t = self.linear_sp(sp_output)
			sp_output_aux = Variable(sp_output_t.data.new(sp_output_t.size(0), sp_output_t.size(1), 1).zero_())
			predict_support = torch.cat((sp_output_aux, sp_output_t), dim=-1).contiguous()

			return predict_support

		for _ in range(self.config.reason_step):
			output_memory = self.memory(output_memory, context_lens, context_mask)

		output_start = torch.cat((output_memory, output), dim=2)
		output_start = self.rnn_start(output_start, context_lens)
		logit1 = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
		output_end = torch.cat((output_memory, output_start), dim=2)
		output_end = self.rnn_end(output_end, context_lens)
		logit2 = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)

		output_type = torch.cat((output, output_end), dim=2)
		output_type = torch.max(self.rnn_type(output_type, context_lens), 1)[0]
		predict_type = self.linear_type(output_type)

		if not return_yp: return logit1, logit2, predict_type

		outer = logit1[:, :, None] + logit2[:, None]
		outer_mask = self.get_output_mask(outer)
		outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
		yp1 = outer.max(dim=2)[0].max(dim=1)[1]
		yp2 = outer.max(dim=1)[0].max(dim=1)[1]
		return logit1, logit2, predict_type, yp1, yp2
