# Model Configs

class ModelConfig:
    
    def __init__(
        self, 
        model_name_or_path, 
        tokenizer_name_or_path = None, 
        ckpt_path=None,  # must be passed if not None
        use_relative_position = False,
        use_abs_position = False,
        max_position_embeddings = None, 
        conv1d_configs = None  # FIXME: how to handle this?
    ) -> None:

        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.ckpt_path = ckpt_path
        self.use_relative_position = use_relative_position
        self.use_abs_position = use_abs_position
        self.max_position_embeddings = max_position_embeddings
        self.conv1d_configs = conv1d_configs
        
        self.cfg = self.return_config(
            model_name_or_path, tokenizer_name_or_path, ckpt_path, 
            use_relative_position, use_abs_position, max_position_embeddings, 
            conv1d_configs
        )

    def return_config(
        self, 
        model_name_or_path, 
        tokenizer_name_or_path, 
        ckpt_path=None,
        use_relative_position = False,
        use_abs_position = False,
        max_position_embeddings = None, 
        conv1d_configs = None
    ):
        """
        just a dummy function to return the config
        """
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path

        ### mamba config
        if "mamba" in model_name_or_path.lower(): 
            
            # 370M model
            if "370" in model_name_or_path.lower():
                # config position embeddings
                if "abs_pos" in model_name_or_path.lower():
                    use_relative_position = True
                elif "rel_pos" in model_name_or_path.lower():
                    use_abs_position = True
                    max_position_embeddings = 16384

                # config kernel sizes
                if "k8" in model_name_or_path.lower():
                    conv1d_configs = {"kernel_sizes": 8}
                elif "k16" in model_name_or_path.lower():
                    conv1d_configs = {"kernel_sizes": 16}
                elif "k32" in model_name_or_path.lower():
                    conv1d_configs = {"kernel_sizes": 32}
                elif "k64" in model_name_or_path.lower():
                    conv1d_configs = {"kernel_sizes": 64}
                elif "km" in model_name_or_path.lower():
                    conv1d_configs = {"kernel_sizes": [[2, 4, 8, 16, 32, 64]]}

                # return mamba config
                return ModelConfig.mamba_config(
                    model_name_or_path = "mamba-370m-hf", 
                    tokenizer_name_or_path = tokenizer_name_or_path, 
                    ckpt_path = ckpt_path, 
                    load_model_state_dict = ckpt_path is not None,
                    use_relative_position = use_relative_position, 
                    use_abs_position = use_abs_position,
                    max_position_embeddings = max_position_embeddings, 
                    conv1d_configs = conv1d_configs,
                ) 

            elif "1_4b" in model_name_or_path.lower():
                return ModelConfig.mamba_config(
                    model_name_or_path = "mamba-1.4b", 
                    tokenizer_name_or_path = tokenizer_name_or_path, 
                    ckpt_path = ckpt_path, 
                    load_model_state_dict = ckpt_path is not None,
                )


        ### deepseek config
        elif "deepseek" in model_name_or_path.lower():
            return ModelConfig.deepseek_config(
                model_name_or_path = "deepseek-coder-1.3b-base", 
                ckpt_path = ckpt_path, 
                load_model_state_dict = ckpt_path is not None,
            )


    @classmethod
    def mamba_config(
        cls, 
        model_name_or_path, 
        tokenizer_name_or_path, 
        ckpt_path=None,
        load_model_state_dict = False,
        use_relative_position = False,
        use_abs_position = False,
        max_position_embeddings = None, 
        conv1d_configs = None
    ):
        mamba_config = {
            "model_name_or_path": model_name_or_path,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "ckpt_path": ckpt_path,
            "load_model_state_dict": load_model_state_dict,
            "use_relative_position": use_relative_position,
            "use_abs_position": use_abs_position,
            "max_position_embeddings": max_position_embeddings,
            "conv1d_configs": conv1d_configs,
        }

        return mamba_config
    

    @classmethod
    def deepseek_config(
        cls,
        model_name_or_path,
        ckpt_path,
        load_model_state_dict,
    ):
        return {
            "model_name_or_path": model_name_or_path,
            "tokenizer_name_or_path": model_name_or_path,
            "load_model_state_dict": load_model_state_dict,
            "ckpt_path": ckpt_path,
        }

