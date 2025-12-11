FROM python:3.12-alpine

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Alpine 使用 apk，速度更快且稳定
# 添加构建依赖（编译某些 Python 包需要）
# 必需依赖说明：
# - gcc, g++: 编译 C/C++ 扩展（cryptography, uvloop, pandas 需要）
# - musl-dev: Alpine 使用 musl libc，编译时需要
# - libffi-dev: cryptography 依赖
# - openssl-dev: cryptography 依赖
# - python3-dev: Python 头文件，编译扩展需要
# - linux-headers: 某些 C 扩展需要
# - make: 编译工具
# 使用 --virtual .build-deps 创建虚拟包组，便于后续删除以减小镜像体积
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    openssl-dev \
    python3-dev \
    make

WORKDIR /app

# 先复制 requirements.txt，利用 Docker 缓存层
# 这样只有在依赖变更时才重新安装
COPY requirements.txt .

# 使用国内 PyPI 源安装依赖
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# 删除构建依赖，减小镜像体积（从 ~500MB 减少到 ~200MB）
# # 只保留运行时需要的库（libffi, openssl 是 cryptography 运行时依赖）
# RUN apk del .build-deps && \
#     apk add --no-cache \
#     libffi \
#     openssl

# 复制项目源代码（在依赖安装后，利用 Docker 缓存层）
COPY . .

# 暴露端口
EXPOSE 8010

# 创建日志目录
RUN mkdir -p /app/logs

# 设置默认命令（使用数组格式，更安全）
# 日志输出到 stdout/stderr，这样 docker logs 可以看到，同时应用日志会写入 logs/ 目录
# 如果需要同时写入文件，可以使用 tee 或日志轮转工具
CMD ["gunicorn", "app:app", \
    "-w", "4", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8010", \
    "--timeout", "300", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--log-level", "info", \
    "--capture-output"]