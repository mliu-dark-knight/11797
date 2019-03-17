import re
import string
from collections import Counter

import torch
from sklearn import metrics

IGNORE_INDEX = -100

RE_D = re.compile('\d')
def has_digit(string):
    return RE_D.search(string)

def prepro(token):
    return token if not has_digit(token) else 'N'


def get_buckets(record_file):
    # datapoints = pickle.load(open(record_file, 'rb'))
    datapoints = torch.load(record_file)
    return [datapoints]

def convert_tokens(eval_file, qa_id, pp1, pp2, p_type):
    answer_dict = {}
    for qid, p1, p2, type in zip(qa_id, pp1, pp2, p_type):
        if type == 0:
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
        elif type == 1:
            answer_dict[str(qid)] = 'yes'
        elif type == 2:
            answer_dict[str(qid)] = 'no'
        elif type == 3:
            answer_dict[str(qid)] = 'noanswer'
        else:
            assert False
    return answer_dict

def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answer"]
        prediction = value
        assert len(ground_truths) == 1
        cur_EM = exact_match_score(prediction, ground_truths[0])
        cur_f1, _, _ = f1_score(prediction, ground_truths[0])
        exact_match += cur_EM
        f1 += cur_f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_sp(sp_true, sp_pred):
    return 100. * metrics.f1_score(sp_true, sp_pred)
