#!/bin/bash  
  
while true; do  
    if nvidia-smi | grep -q 'No running processes found'; then  
        cd /zecheng/GrabGPU/  
        ./gg 30 240 0,1,2,3,4,5,6,7  
        break  
    else  
        gpustat
        date  
        sleep 30
    fi  
done  
