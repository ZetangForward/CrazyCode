class OptimizerConfig:
    def __init__(self, opt_name, train_step, warmup_step) -> None:
        self.opt_name = opt_name
        self.train_step = train_step
        self.warmup_step = warmup_step
        
        self.cfg = self.return_config(opt_name, train_step, warmup_step)

    def return_config(self, opt_name, train_step=20000, warmup_step=2000):
        if "adawm" in opt_name.lower():   
            return OptimizerConfig.adamw_config(train_step, warmup_step) 
        else:
            ...


    @classmethod
    def adamw_config(cls, num_training_steps, warmup_steps):
        adamw_config = {
            "optimizer_type": "adamw",
            "lr": 5e-5,
            "beta_1": 0.9,
            "beta_2": 0.95,
            "num_training_steps": num_training_steps,
            "warmup_steps": warmup_steps,
            "peak_lr": 0.0002,
            "last_lr": 0.00001,
        }

        return adamw_config

