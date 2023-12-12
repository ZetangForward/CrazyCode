#!/bin/bash  
###
 # @Author: ZetangForward 1
 # @Date: 2023-12-12 04:05:54
 # @LastEditors: ZetangForward 1
 # @LastEditTime: 2023-12-12 04:20:42
 # @FilePath: /Detox-CoT/ZipCode/GrabGPU/auto_grab_gpu.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
  
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
