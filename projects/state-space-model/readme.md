Debug Config



```json
{
    "version": "0.2.0",
    "configurations": [
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
