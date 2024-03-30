class PlatformConfig:
    def __init__(self, platform_name='amax_a100') -> None:
        self.platform_name = platform_name
        self.return_config(platform_name)

    def return_config(self, platform_name):
        if "amax" in platform_name.lower():  
            if "a100" in platform_name.lower():
                return PlatformConfig.amax_a100()
            
            elif "3090" in platform_name.lower():
                return PlatformConfig.amax_3090()
            
        elif "langchao" in platform_name.lower():
            return PlatformConfig.langchao()


    @classmethod
    def langchao(cls):
        return {
            "name":"langchao_suda",
            "hf_model_path": "/public/home/ljt/hf_models",
            "dataset_path": "/public/home/ljt/tzc/data",
            "exp_path": "/public/home/ljt/tzc/ckpt",
            "result_path": "/public/home/ljt/tzc/data/evaluation",
        }

    @classmethod
    def amax_a100(cls):
        return {
            "name":"amax_a100",
            "hf_model_path": "/nvme/hf_models",
            "dataset_path": "/nvme/zecheng/data",
            "exp_path": "/nvme/zecheng/ckpt",
            "result_path": "/nvme/zecheng/evaluation",
        }

    @classmethod
    def amax_3090(cls):
        return {
            "name":"amax_3090",
            "hf_model_path": "/opt/data/private/hf_models",
            "dataset_path": "/opt/data/private/zecheng/data",
            "exp_path": "/opt/data/private/zecheng/ckpt",
            "result_path": "/opt/data/private/zecheng/evaluation",
        }


