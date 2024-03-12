import sys
import os
sys.path.append(os.getcwd())
from modelzipper.tutils import *
from rouge_score import rouge_scorer
import tensor_parallel as tp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from utils import get_model_tokenizer, get_model_tokenizer_simple
from argparse import ArgumentParser


class Evaluator:

    def __init__(self, root_dir, fpath, task, tokenizer_name_or_path, save_evaluation_path=None, save_gen_res=True, **kwargs) -> None:
        self.task = task
        self.root_dir = root_dir
        self.predictions = auto_read_data(fpath)
        self.tokenizer, _ = get_model_tokenizer_simple(root_dir, tokenizer_name_or_path)
        self.spe_cfg = kwargs
        self.begin_fn(task, save_evaluation_path, save_gen_res)

    
    def begin_fn(self, task, save_evaluation_path, save_gen_res):
        if "passkey" in task.lower():
            self.eval_passkey_search(save_evaluation_path, save_gen_res) 


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



if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--root_dir", type=str, default="/nvme/hf_models")
    argparse.add_argument("--fpath", type=str, default="/nvme/zecheng/evaluation/passkey_search/mamba-1_4b/version_2/results/predictions.pkl")
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
        root_dir=args.root_dir, fpath=args.fpath, task=args.task,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        value=args.value, save_evaluation_path=args.save_evaluation_path,
        save_gen_res=args.save_gen_res,
    )
