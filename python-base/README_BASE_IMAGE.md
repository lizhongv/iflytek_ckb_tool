# 通用 Python 基础镜像使用指南

这是一个通用的 Python 3.12 基础镜像，预装了常用的 Python 包和工具，可以用于快速运行各种 Python 应用。

## 方式一：基础镜像使用

```bash
# 1. 构建基础镜像
# 确保 requirements.base.txt 和 docker-entrypoint.sh 在同一目录
docker build -f Dockerfile.base -t python-base:3.12 .
# Dockerfile.base 基础镜像定义
# requirements.base.txt 基础依赖包
# docker-entrypoint.sh 通用启动脚本
# 暴漏常用端口（8000, 8010, 8080, 8888, 5000）

# 2. 直接使用基础镜像运行代码（推荐用于开发/测试）
#（1） 运行 Python 脚本
docker run --rm -v $(pwd):/app python-base:3.12 python your_script.py
#（2） 运行 FastAPI 应用（uvicorn）
docker run --rm -v $(pwd):/app -p 8000:8000 \
python-base:3.12 uvicorn app:app --host 0.0.0.0 --port 8000
#（3） 运行 FastAPI 应用（gunicorn）
docker run --rm -v $(pwd):/app -p 8010:8010 \
python-base:3.12 gunicorn app:app -w 4 \
-k uvicorn.workers.UvicornWorker \
--bind 0.0.0.0:8010
#（4） 交互式 shell
docker run -it --rm -v $(pwd):/app python-base:3.12 /bin/bash
#（5）查看已安装的包
docker run --rm python-base:3.12 pip list
```


## 方式二：基于基础镜像构建应用镜像（推荐用于生产）

```bash
# 1. 创建应用 Dockerfile
# 如 Dockerfile.app.example

# 2. 构建应用镜像
docker build -f Dockerfile.app.example -t my-app:latest .

# 3. 运行应用容器
docker run -d --name my-app -p 8010:8010 my-app:latest
```

## 方式三：使用 docker-compose 构建应用镜像

```yaml
version: '3.8'

services:
  app1:
    build:
      context: .
      dockerfile: Dockerfile.app.example
    ports:
      - "8010:8010"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
```

```bash
# 启动所有服务（后台运行）
docker-compose up -d

# 启动并查看日志
docker-compose up

# 只启动 app1 服务
docker-compose up app1
```


## 目录结构

```
/app/
├── logs/      # 日志目录
├── data/      # 数据目录
└── config/    # 配置目录
```

## 环境变量

- `PYTHONPATH=/app` - Python 模块搜索路径
- `PYTHONUNBUFFERED=1` - 禁用 Python 输出缓冲
- `PYTHONDONTWRITEBYTECODE=1` - 不生成 .pyc 文件

## 暴露的端口

- 8000 - 常用 Web 服务端口
- 8010 - 常用 API 端口
- 8080 - 备用 Web 端口
- 8888 - Jupyter/开发端口
- 5000 - Flask 默认端口

## 示例：运行不同的应用

### 示例 1：运行 FastAPI 应用

```bash
# 假设你的应用在 ./myapp 目录
docker run -d --name fastapi-app \
  -v $(pwd)/myapp:/app \
  -p 8000:8000 \
  python-base:3.12 \
  uvicorn main:app --host 0.0.0.0 --port 8000
```

### 示例 2：运行数据处理脚本

```bash
docker run --rm \
  -v $(pwd)/scripts:/app \
  -v $(pwd)/data:/app/data \
  python-base:3.12 \
  python process_data.py
```

### 示例 3：运行定时任务

```bash
docker run -d --name scheduler \
  -v $(pwd)/tasks:/app \
  python-base:3.12 \
  python scheduler.py
```


## 注意事项

1. 基础镜像体积较大（~500MB），但包含常用包
2. 如果应用需要特殊依赖，可以在应用 Dockerfile 中额外安装
3. 生产环境建议基于基础镜像构建应用镜像，而不是直接挂载代码
4. 定期更新基础镜像以获取安全补丁

## 更新基础镜像

```bash
# 重新构建基础镜像
docker build -f Dockerfile.base -t python-base:3.12 .

# 更新所有基于此镜像的应用
docker-compose build --no-cache
```

## 故障排查

```bash
# 1. 查看基础镜像信息
docker inspect python-base:3.12

# 2. 测试基础镜像
docker run --rm python-base:3.12 python -c "import fastapi; print('FastAPI OK')"

# 3. 查看已安装的包
docker run --rm python-base:3.12 pip list
```


