#!/bin/bash
# ============================================================================
# 通用 Docker 入口脚本
# 支持多种启动方式：
# 1. 直接运行 Python 文件：docker run image python app.py
# 2. 运行 uvicorn：docker run image uvicorn app:app --host 0.0.0.0 --port 8000
# 3. 运行 gunicorn：docker run image gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
# 4. 交互式 shell：docker run -it image /bin/bash
# ============================================================================

set -e

# 如果第一个参数是命令，直接执行
if [ "$1" = "python" ] || [ "$1" = "uvicorn" ] || [ "$1" = "gunicorn" ] || [ "$1" = "/bin/bash" ] || [ "$1" = "/bin/sh" ]; then
    exec "$@"
fi

# 如果第一个参数是 Python 文件，使用 python 运行
if [ -f "$1" ] && [[ "$1" == *.py ]]; then
    exec python "$@"
fi

# 如果提供了自定义命令，执行它
if [ $# -gt 0 ]; then
    exec "$@"
fi

# 默认行为：显示帮助信息
cat << EOF
通用 Python 基础镜像使用说明：

1. 运行 Python 文件：
   docker run -v \$(pwd):/app image python your_script.py

2. 运行 FastAPI 应用（uvicorn）：
   docker run -v \$(pwd):/app -p 8000:8000 image uvicorn app:app --host 0.0.0.0 --port 8000

3. 运行 FastAPI 应用（gunicorn）：
   docker run -v \$(pwd):/app -p 8010:8010 image gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8010

4. 交互式 shell：
   docker run -it -v \$(pwd):/app image /bin/bash

5. 查看已安装的包：
   docker run image pip list

环境变量：
  - PYTHONPATH=/app
  - 工作目录：/app
  - 日志目录：/app/logs
  - 数据目录：/app/data
  - 配置目录：/app/config

暴露的端口：8000, 8010, 8080, 8888, 5000
EOF

exec "$@"

