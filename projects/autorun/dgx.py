import sys
import subprocess

def ssh_to_dgx(node_number):
    node_name = f"dgx-{node_number}"
    command = f"ssh {node_name} 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'"
    ssh_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = ssh_process.communicate()

    if stderr:
        print(stderr.decode().strip())
        return None
    gpu_info = stdout.decode().strip().split('\n')
    print("{} : {}".format(node_name,gpu_info))
    # free_gpus = [gpu.split(',')[0] for gpu in gpu_info if int(gpu.split(',')[1].strip().split(' ')[0]) > 70000]
    return 0
    # import os
    # import psutil
    # import xml.etree.ElementTree as ET
    # import time
    # from pprint import pprint
    # from datetime import timedelta, datetime

    # def execute_command(command_str):
    #     p = os.popen(command_str)
    #     text = p.read()
    #     p.close()
    #     return text

    # # command_str = 'nvidia-smi dmon -c 1'
    # # pwr：功耗，temp：温度，sm：流处理器，mem：显存，enc：编码资源，dec：解码资源，mclk：显存频率，pclk：处理器频率
    # # # gpu   pwr  temp    sm   mem   enc   dec  mclk  pclk
    # # Idx     W     C     %     %     %     %   MHz   MHz
    # #     0   112    87    19     7     0     0  5005  1480
    # #     1   177    83    78     8     0     0  5005  1809
    # #     2    93    81    31    10     0     0  5005  1809
    # #     3   150    83    93    54     0     0  5005  1784

    # def get_process_info(pid):
    #     p = psutil.Process(pid)
    # #     print(p.as_dict())
    #     return p.as_dict()

    # def get_gpu_info():
    #     command_str = 'nvidia-smi -q -x'
    #     # 查询所有GPU的当前详细信息并将查询的信息以xml的形式输出
    #     gpus_info_dict = {}
    #     text = execute_command(command_str)
    #     root = ET.fromstring(text)
    #     for i, gpu_info in enumerate(root.findall('gpu')):
    #         gpu_info_dict = {}
    #         fb_memory_usage = gpu_info.find('fb_memory_usage')
    #         total = fb_memory_usage.find('total').text
    #         used = fb_memory_usage.find('used').text
    #         rate = int(used.split(' ')[0]) / int(total.split(' ')[0])
    #         gpu_info_dict.update({'gpu utilization':rate})

    #         gpus_info_dict.update({f'gpu:{i}':gpu_info_dict})
    #     return gpus_info_dict

    # pprint(get_gpu_info())

    return free_gpus

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <node_number>")
        sys.exit(1)
    import psutil
    
    node_number = sys.argv[1]
    free_gpus = ssh_to_dgx(node_number)
    # if free_gpus:
    #     print(f"Node dgx-{node_number} has the following free GPUs: {', '.join(free_gpus)}")
    # else:
    #     print(f"Failed to get GPU information for node dgx-{node_number}.")
