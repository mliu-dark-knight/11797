import numpy as np

from common.model import *


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
	def __init__(self, config, word_mat, char_mat):
		super().__init__()
		self.config = config
		self.word_dim = config.glove_dim
		self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
		self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
		self.word_emb.weight.requires_grad = False
		self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
		self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

		self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
		self.char_hidden = config.char_hidden
		self.hidden = config.hidden

		self.rnn = EncoderRNN(config.char_hidden + self.word_dim, config.hidden, 1, True, True, 1 - config.keep_prob,
		                      False)

		self.qc_att = BiAttention(config.hidden * 2, 1 - config.keep_prob)
		self.linear_1 = nn.Sequential(
			nn.Linear(config.hidden * 8, config.hidden),
			nn.ReLU()
		)

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

	def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping,
	            end_mapping, stage='locate', return_yp=False):
		para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(
			2), context_idxs.size(0)

		context_mask = (context_idxs > 0).float()
		ques_mask = (ques_idxs > 0).float()

		context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size,
		                                                                                    -1)
		ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

		context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
		ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

		context_word = self.word_emb(context_idxs)
		ques_word = self.word_emb(ques_idxs)

		context_output = torch.cat((context_word, context_ch), dim=2)
		ques_output = torch.cat((ques_word, ques_ch), dim=2)

		context_output = self.rnn(context_output, context_lens)
		ques_output = self.rnn(ques_output)

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
