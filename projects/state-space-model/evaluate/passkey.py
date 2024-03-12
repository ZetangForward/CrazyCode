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


class Evaluator:

    def __init__(self, root_dir, fpath, task, tokenizer_name_or_path, **kwargs) -> None:
        self.task = task
        self.root_dir = root_dir
        self.predictions = auto_read_data(fpath)
        self.tokenizer, _ = get_model_tokenizer_simple(root_dir, tokenizer_name_or_path)
        self.spe_cfg = kwargs

        self.begin_fn(task)

        
    
    def begin_fn(self, task):
        if "passkey" in task.lower():
            self.eval_passkey_search() 

    def eval_passkey_search(self):
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
            context_length = item['ctx_length']
            pred = pred[context_length:]
            str_pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            score = scorer.score(needle, str_pred)['rouge1'].fmeasure*10
            depth, context_length = item['depth'], item['ctx_length']
            results.append({
                'depth': depth, 'ctx_length': context_length, 'score': score
            })

        print_c(f"passkey search evaluation finished, total {len(results)} instances", "yellow")

        self.visualize_passkey_search(results)
        


    def visualize_passkey_search(self, results):
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

        save_path = "evaluate/passkey_search_results.png"
        print("saving at %s" % save_path)
        plt.savefig(save_path, dpi=150)



if __name__ == "__main__":
    evaluator = Evaluator(
        root_dir="/nvme/hf_models",
        fpath="/nvme/zecheng/evaluation/passkey_search/deepseek-1_3b/version_1/results/predictions.pkl",
        task="passkey_search",
        tokenizer_name_or_path="EleutherAI/gpt-neox-20b",
        value="eat a sandwich and sit in Dolores Park on a sunny day."
    )
    # evaluator.eval_passkey_search()