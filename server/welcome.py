import os
import subprocess

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')

# 激活conda环境
print("正在激活conda环境...")
conda_activate_command = "conda activate zecheng"
output, error = run_command(conda_activate_command)
print(output)
if error:
    print(f"错误信息: {error}")

# 加载第一个模块
print("正在加载compiler/gcc/10.3.0模块...")
module_load_command1 = "module load compiler/gcc/10.3.0"
output, error = run_command(module_load_command1)
print(output)
if error:
    print(f"错误信息: {error}")

# 加载第二个模块
print("正在加载compiler/cmake/3.20.1模块...")
module_load_command2 = "module load compiler/cmake/3.20.1"
output, error = run_command(module_load_command2)
print(output)
if error:
    print(f"错误信息: {error}")

print("环境设置完成！")
