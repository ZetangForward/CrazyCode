from modelzipper.tutils import *
from rouge_score import rouge_scorer
import tensor_parallel as tp

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


class Evaluator:

    def __init__(self, fpath, task) -> None:
        self.task = task
        self.prediction = auto_read_data(fpath)
        

def eval_passkey_search():
    pass