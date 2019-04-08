from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

BIG_INT = 1e8
SMALL_FLOAT = 1e-2
IGNORE_INDEX = -1
FULL_BATCH_KEY = 'full_batch'
CONTEXT_IDXS_KEY = 'context_idxs'
QUES_IDXS_KEY = 'ques_idxs'
CONTEXT_QUES_IDXS_KEY = 'context_ques_idxs'
CONTEXT_QUES_MASKS_KEY = 'context_ques_masks'
CONTEXT_QUES_SEGMENTS_KEY = 'context_ques_segments'
ANSWER_MASKS_KEY = 'answer_masks'
QUES_SIZE_KEY = 'ques_size'
ID_KEY = 'id'
IDS_KEY = 'ids'
IS_SUPPORT_KEY = 'is_support'
ALL_MAPPING_KEY = 'all_mapping'
Y1_KEY = 'y1'
Y1_FLAT_KEY = 'y1_flat'
Y2_KEY = 'y2'
Y2_FLAT_KEY = 'y2_flat'
ORIG_IDXS = 'orig_idxs'
ORIG_IDXS_R = 'orig_idxs_r'
Q_TYPE_KEY = 'q_type'
START_END_FACTS_KEY = 'start_end_facts'
SEP = '[SEP]'
CLS = '[CLS]'
UNK = '[UNK]'
SEP_IDX = tokenizer.vocab[SEP]
CLS_IDX = tokenizer.vocab[CLS]
UNK_IDX = tokenizer.vocab[UNK]
PARA_LIMIT = 144
QUES_LIMIT = 45
ANS_LIMIT = 8
BERT_HIDDEN = 768
