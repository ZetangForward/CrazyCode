#!/bin/bash
SESSION_NAME="SUWA"
ROWS=4
COLS=4
WINDOWS=1
 
# 使用循环生成 commands 列表
commands=()
for i in $(seq 0 15); do
  commands+=("echo worker-$i && ssh worker-$i")
  # commands+=("echo worker-$i && ssh -t worker-$i 'python'")
done
 
create_window() {
  local window_index=$1
  local cmd_index=$((window_index * ROWS * COLS))
 
  if [ $window_index -eq 0 ]; then
    cmd_index=$((cmd_index + 1))
  else
    tmux send-keys -t $SESSION_NAME:$window_index "${commands[$cmd_index]}" Enter
    cmd_index=$((cmd_index + 1))
  fi
 
  # 创建水平（横向）窗格
  for i in $(seq 1 $((COLS - 1))); do
    tmux split-window -v -t $SESSION_NAME:$window_index "${commands[$cmd_index]}"
    tmux select-layout -t $SESSION_NAME:$window_index tiled
    cmd_index=$((cmd_index + 1))
  done
 
  # 切换到每个横向窗格并创建垂直（竖直）窗格
  for i in $(seq 0 $((COLS - 1))); do
    tmux select-pane -t $SESSION_NAME:$window_index.$((i * ROWS))
    for j in $(seq 1 $((ROWS - 1))); do
      tmux split-window -h -t $SESSION_NAME:$window_index "${commands[$cmd_index]}"
      tmux select-layout -t $SESSION_NAME:$window_index tiled
      cmd_index=$((cmd_index + 1))
    done
  done
  tmux setw -t $SESSION_NAME:$window_index synchronize-panes on
}
 
# 创建一个新的 tmux 会话，使用第一个命令作为初始命令
tmux new-session -d -s $SESSION_NAME "${commands[0]}"
 
# 创建第一个 4x4 窗格窗口
create_window 0
 
# 创建额外的窗口
for i in $(seq 1 $((WINDOWS - 1))); do
  tmux new-window -t $SESSION_NAME:$i
  tmux select-window -t $SESSION_NAME:$i
  create_window $i
done
 
# 附加到新创建的 tmux 会话
tmux attach-session -t $SESSION_NAME