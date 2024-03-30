Debug Config



```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "mamba_debug_train",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/train_dev2.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
            "args": [
                "--model_name_or_path", "mamba-370m-k8",
                "--platform_name", "amax_a100",
                "--experiment_name", "test",
                "--ckpt_path", "/nvme/zecheng/ckpt/h_800/ckpt/slimpajama/mamba_370m_big_kernel-k8/checkpoints/last.ckpt/model.bin"
            ]
        },
        {
            "name": "mamba_analysis",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/analysis.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
        {
            "name": "mamba_debug_train",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/train.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
        {
            "name": "mamba_analysis",
            "type": "python",
            "request": "launch",
            "program": "/nvme/zecheng/modelzipper/projects/state-space-model/src/analysis.py",
            "console": "integratedTerminal",
            "cwd": "/nvme/zecheng/modelzipper/projects/state-space-model",
            "env": {
                "CUDA_VISIBLE_DEVICES": "7"
            },
            "python": "/home/amax/anaconda3/envs/zecheng/bin/python",
            "justMyCode": false,
        },
    ]
}

```
