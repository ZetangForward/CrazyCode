import sys
import os
sys.path.append(os.getcwd())
from modelzipper.tutils import *
# from rouge_score import rouge_scorer
import tensor_parallel as tp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from utils import get_model_tokenizer, get_model_tokenizer_simple
from argparse import ArgumentParser

from metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
)

longbench_dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
}

class Evaluator:

    def __init__(self, root_dir, fpath, data_path, task, tokenizer_name_or_path, subtask=None, save_evaluation_path=None, save_gen_res=True, **kwargs) -> None:
        self.task = task
        self.subtask = subtask
        self.fpath = fpath
        self.data_path = data_path
        self.root_dir = root_dir
        self.predictions = auto_read_data(fpath) if "bench" not in task else None
        # self.tokenizer, _ = get_model_tokenizer_simple(root_dir, tokenizer_name_or_path)
        self.spe_cfg = kwargs
        self.begin_fn(task, save_evaluation_path, save_gen_res)

    
    def begin_fn(self, task, save_evaluation_path, save_gen_res):
        if "passkey" in task.lower():
            self.eval_passkey_search(save_evaluation_path, save_gen_res) 
        
        if "ar" in task.lower():
            self.eval_mqar(save_evaluation_path, save_gen_res)

        if "longbench" in task.lower():
            self.eval_longbench(save_evaluation_path, save_gen_res)


    def eval_passkey_search(self, save_evaluation_path, save_gen_res=True):
        """
        dict_keys = ['attention_mask', 'depth', 'key', 'value', 'ctx_length', 'predictions']
        """
        assert "value" in self.spe_cfg, "value is required for passkey search"
        needle = self.spe_cfg['value']
        print_c("initiating passkey search evaluation ...", "yellow")
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        results = []

        for item in self.predictions:
            pred = item['predictions'].squeeze(0)
            if 'attention_mask' in item:
                real_context_length = item['attention_mask'].sum().item()
            else:
                real_context_length = item['real_length']
            pred = pred[real_context_length:]
            str_pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            score = scorer.score(needle, str_pred)['rouge1'].fmeasure*10
            depth, context_length = item['depth'], item['ctx_length']
            results.append({
                'depth': round(depth, 2), 'ctx_length': context_length, 'score': score, 'pred': str_pred,
            })

        print_c(f"passkey search evaluation finished, total {len(results)} instances", "yellow")

        if save_gen_res:
            save_path = os.path.join(save_evaluation_path, "generation.jsonl")
            print_c(f"saving at {save_path}", "yellow")
            auto_save_data(results, save_path)

        self.visualize_passkey_search(results, save_evaluation_path)

        
    def visualize_passkey_search(self, results, save_evaluation_path):
        """
            results: dict [ depth, ctx_length, score ]
        """
        # Creating a DataFrame
        df = pd.DataFrame(results)
        df['depth'] = df['depth'].round(2)

        pivot_table = pd.pivot_table(df, values='score', index=['depth', 'ctx_length'], aggfunc='mean').reset_index() # This will aggregate
        pivot_table = pivot_table.pivot(index="depth", columns="ctx_length", values="score") # This will turn into a proper pivot

        # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
        
        # Create the heatmap with better aesthetics
        f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
        heatmap = sns.heatmap(
            pivot_table,
            vmin=0, 
            vmax=1,
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor='grey',
            linestyle='--'
        )

        title_font = {
            'fontsize': 16,
            'fontweight': 'bold',
            # 'fontname': 'Arial'
        }

        label_font = {
            'fontsize': 16,
            # 'fontweight': 'bold',
            # 'fontname': 'Arial'
        }

        x_values = df['ctx_length'].unique()
        x_ticks = x_values[5::6]  # take every 5 steps
        steps = list(range(6, len(x_values), 6))

        # 设置横坐标的位置和标签
        heatmap.set_xticks(steps)
        heatmap.set_xticklabels(x_ticks, rotation=0)

        ax = heatmap.get_figure().get_axes()[0]

        for j in steps:
            ax.axvline(x=j, color='black', linestyle=':', linewidth=1.5)

        ## More aesthetics
        # plt.title('Passkey Search Results', **title_font)  # Adds a title
        plt.xlabel('Context Length', **label_font)  # X-axis label
        plt.ylabel('Passkey Depth', **label_font)  # Y-axis label
        plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
        plt.tight_layout()  # Fits everything neatly into the figure area

        save_path = os.path.join(save_evaluation_path, "passkey_search_results.png")
        print("saving at %s" % save_path)
        plt.savefig(save_path, dpi=150)

    def eval_mqar(self, save_evaluation_path, save_gen_res=True):
        

        # if not self.predictions[0].get('labels') and self.data_path is not None:
        #     data = auto_read_data(self.data_path)

        total_number = len(self.predictions)
        correct_number = 0

        for i in range(len(self.predictions)):
            item = self.predictions[i]
            pred = item['predictions'].squeeze(0)
            # if not item.get('labels'):
            #     label = data[i]['label']
            # else:
            label = item['labels'].squeeze(0)
            target_idx = label!=-100
            pred_value = pred[target_idx]
            label_value = label[target_idx]
            correct_predictions = (pred_value == label_value).sum()
       
            correct_number += ( correct_predictions==target_idx.sum() )

        # if save_gen_res:
        #     save_path = os.path.join(save_evaluation_path, "generation.jsonl")
        #     print_c(f"saving at {save_path}", "yellow")
        #     auto_save_data(results, save_path)
        if save_evaluation_path:
            # auto_save_data([str(correct_number/total_number * 100)], save_evaluation_path)
            result = str(correct_number/total_number * 100)
            with open(save_evaluation_path,'a+') as f:
                f.write(result+"\n")

        print(self.task, save_evaluation_path, correct_number/total_number * 100)

    
    def eval_longbench(self, save_evaluation_path, save_gen_res=True):

        scores = dict()
        if subtask is None:
            sub_tasks = ["narrativeqa","qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", \
                        "musique", "gov_report", "qmsum", "multi_news", "trec", \
                        "triviaqa", "samsum", "passage_count", "passage_retrieval_en"]
        else:
            sub_tasks = [ self.subtask ] 


        for subtask in sub_tasks :
        # for subtask in [ "passage_count", "passage_retrieval_en"] :
        #     import pdb;pdb.set_trace()
            total_score = 0
            self.predictions = auto_read_data(self.fpath)
            if subtask == "trec": 
                trec_path = os.join(self.data_path, "/trec.jsonl")
                all_class = auto_read_data(trec_path)[0]['all_classes']
            else:
                all_class = None
            for item in self.predictions:
                prediction = item["answers"]
                ground_truths = item["labels"][0]
                score = 0
                if subtask in ["trec", "triviaqa", "samsum", "lsht"]:
                    prediction = prediction.lstrip('\n').split('\n')[0]
                for ground_truth in ground_truths:
                    score = max(score, longbench_dataset2metric[subtask](prediction, ground_truth, all_classes=all_class))
                total_score += score
            scores[subtask] = str(round(100 * total_score / len(self.predictions), 2))
            print(subtask, round(100 * total_score / len(self.predictions), 2), longbench_dataset2metric[subtask])
        auto_save_data([scores],save_evaluation_path+"eval.jsonl")

if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--root_dir", type=str, default="/nvme/hf_models")
    argparse.add_argument("--fpath", type=str, default="/nvme/zecheng/evaluation/passkey_search/mamba-1_4b/version_2/results/predictions.pkl")
    argparse.add_argument("--data_path", type=str, default=None)   
    argparse.add_argument("--task", type=str, default="passkey_search")
    argparse.add_argument("--tokenizer_name_or_path", type=str, default="EleutherAI/gpt-neox-20b")
    argparse.add_argument("--value", type=str, default="eat a sandwich and sit in Dolores Park on a sunny day.")
    argparse.add_argument("--save_evaluation_path", type=str, default=None)
    argparse.add_argument("--save_gen_res", type=bool, default=True)

    args = argparse.parse_args()

    if args.save_evaluation_path is None:
        args.save_evaluation_path = os.path.dirname(args.fpath)

    print_c(f"args: {args}", "yellow")

    evaluator = Evaluator(
        root_dir=args.root_dir, fpath=args.fpath, data_path=args.data_path, task=args.task,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        value=args.value, save_evaluation_path=args.save_evaluation_path,
        save_gen_res=args.save_gen_res,
    )
