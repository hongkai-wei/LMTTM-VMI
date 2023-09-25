@echo on

rem 激活环境 env1
call conda activate env1

rem 切换到 F 盘
f:

rem 进入目录 F:\git_ttm\logger
cd /d F:\git_ttm\logger
start "" "http://localhost:6006/"
rem 启动 TensorBoard
tensorboard --logdir=.

