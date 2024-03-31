from modelzipper.tutils import *

class Model:
    def __init__(
        self,
        model_name_or_path = "mamba-370m-hf",
        tokenizer_name_or_path = "mamba-370m-hf",
        ckpt_path = "mamba-370m-hf/pytorch_model.bin",
        use_custom_module = True,
        load_model_state_dict = False,
        use_relative_position = False,
        use_abs_position = False,
        max_position_embeddings = 16384,
        custom_conv1d = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.ckpt_path = ckpt_path
        use_custom_module = True
        load_model_state_dict = False
        use_relative_position = False
        use_abs_position = False
        max_position_embeddings = 16384
        custom_conv1d = True
        class Conv1d_configs:
            kernel_sizes = [2, 4, 8, 16, 32, 64]
        conv1d_configs = Conv1d_configs()


class Model:
    model_name_or_path = "mamba-370m-hf"
    tokenizer_name_or_path = "mamba-370m-hf"
    ckpt_path = "mamba-370m-hf/pytorch_model.bin"
    use_custom_module = True
    load_model_state_dict = False
    use_relative_position = False
    use_abs_position = False
    max_position_embeddings = 16384
    custom_conv1d = True
    class Conv1d_configs:
        kernel_sizes = [2, 4, 8, 16, 32, 64]
    conv1d_configs = Conv1d_configs()

class platform:
    name = "光年之外 H800"
    hf_model_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/hf_models"
    dataset_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/data"
    exp_path = "/aifs4su/ziliwang/txw/InternLM/zecheng"
    result_path = "/aifs4su/ziliwang/txw/InternLM/zecheng/evaluation"

class task:
    class Dataset: 
        data_path = "slim_pajama_chunk1/processed_data"
        processed_data_path = ""
        split = ""
        module = "custom_dataset.slimpajama"
        class_name = 'Slimpajama'
        type = "hf"
        max_seq_length = 2048
        nworkers = 8
        train_batch_size = 4
        val_batch_size = 1
        pin_memory = True
        inference_mode = False
        cluster_batch = False
    dataset = Dataset()
    other_cfgs = ""

class experiment:
    seed = 27
    model_save_dir = "simplepajama-mamba_370m_multi-multi"
    save_top_k = 5
    monitor_metric = "train_lm_loss"
    weight_decay = 0.1
    eps = 0.001
    every_n_train_steps = 2000
    accumulate_grad_batches = 2
    use_deepspeed = True
    debug = False
    hf_trainer = True
    low_rank_train = False
    device_num = 8
    node_num = 4

class Optimizer:
    optimizer_type = "adamw"
    lr = 5e-5
    beta_1 = 0.9
    beta_2 = 0.95
    num_training_steps = 20000
    warmup_steps = 2000
    peak_lr = 0.0002
    last_lr = 0.00001

class lr_scheduler:
    scheduler_type = "get_cosine_schedule_with_warmup"
    warmup_steps = 0

class Configs:
    model_name = "mamba_370m_multi"
    mark = model_name + "multi" 
    state = "train"
    exp_task = "simplepajama"
    task = task()
    platform = platform()
    model = Model()
    experiment = experiment()
    optimizer = Optimizer()
    lr_scheduler = lr_scheduler()