#!/bin/bash  
  
while true; do  
    # 使用nvidia-smi命令检查GPU状态  
    if nvidia-smi | grep -q 'No running processes found'; then  
        cd /zecheng/GrabGPU/  
        ./gg 30 240 0,1,2,3,4,5,6,7  
        break  
    else  
        # 如果GPU不空闲，打印当前时间并休眠5分钟  
        gpustat
        date  
        sleep 5  
    fi  
done  
