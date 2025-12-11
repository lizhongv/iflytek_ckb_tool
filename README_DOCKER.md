
# 直接构建新镜像

## 方法一：采用 Dockerfile 构建

```bash
cd /root/zhongli2/iflytek_ckb_tool

# 1、构建镜像
docker build -t ckb-qa-tool:latest .
# 查看构建的镜像
docker images | grep ckb-qa-tool

# 2、运行容器
docker run -d \
  --name ckb-qa-tool \
  -p 8010:8010 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/batch_config.yaml:/app/batch_config.yaml:ro \
  ckb-qa-tool:latest

# 3、查看日志
docker logs -f ckb-qa-tool
```


## 方法二：采用 docker-compose 

```bash
cd /root/zhongli2/iflytek_ckb_tool
# 1、构建镜像（首次运行）
docker-compose build
# 2、启动服务
docker-compose up -d
# 3、查看日志
docker-compose logs -f
docker-compose logs -f ckb-qa-tool
docker-compose logs --tail=100 ckb-qa-tool
# 4、停止服务
docker-compose stop
# 5、重启服务
docker-compose restart
# 6、进入容器
docker-compose exec ckb-qa-tool bash
# 7、查看状态
docker-compose ps
# 8、停止并删除容器
docker-compose down
# 9、重新构建镜像
docker-compose build --no-cache
docker-compose up -d
# 10、查看资源使用情况
docker stats ckb-qa-tool
```

## 方法三：利用启动脚本构建

```bash
cd /root/zhongli2/iflytek_ckb_tool
# 1、启动服务（自动构建镜像）
./docker-run.sh start
# 2、查看日志
./docker-run.sh logs
# 3、停止服务
./docker-run.sh stop
# 4. 重启服务
./docker-run.sh restart
# 5、进入容器
./docker-run.sh shell
# 6、查看状态
./docker-run.sh status
```

## 验证部署

```bash
# 1. 健康检查
curl http://localhost:8010/health
# 2. 查看 API 信息
curl http://localhost:8010/
```

## 带GPU支持（可选）
如果项目需要使用 GPU（T4 卡），需要：

### 1. 修改 docker-compose.yml
取消注释 GPU 相关配置：

```yaml
services:
  ckb-qa-tool:
    # ... 其他配置 ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. 使用 Docker 命令运行

```bash
docker run -d \
  --name ckb-qa-tool \
  --gpus all \
  -p 8010:8010 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ckb-qa-tool:latest
```

### 3. 验证 GPU 访问

```bash
# 进入容器
docker exec -it ckb-qa-tool bash

# 在容器内检查 GPU
nvidia-smi
```



# 利用基础镜像来构建

## 第一步：构建基础镜像

```bash
# 方式一：使用构建脚本（推荐）
./build-base-image.sh

# 方式二：直接使用 docker build
docker build -f Dockerfile.base -t python-base:3.12 .
```

## 第二步 使用基础镜像运行应用

### 场景 1：运行现有的 CKB Tool 应用

```bash
# 方式 A：直接挂载代码运行（开发/测试）
docker run -d --name ckb-tool \
  -v $(pwd):/app \
  -p 8010:8010 \
  python-base:3.12 \
  gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8010 --access-logfile - --error-logfile -

# 方式 B：基于基础镜像构建应用镜像（生产）
# 1. 创建 Dockerfile
cat > Dockerfile << EOF
FROM python-base:3.12
COPY . /app/
WORKDIR /app
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8010"]
EOF

# 2. 构建并运行
docker build -t ckb-tool:latest .
docker run -d --name ckb-tool -p 8010:8010 ckb-tool:latest
```

### 场景 2：运行新的 Python 脚本

```bash
# 创建测试脚本
cat > test.py << EOF
import fastapi
import pandas as pd
print("FastAPI version:", fastapi.__version__)
print("Pandas version:", pd.__version__)
print("All packages loaded successfully!")
EOF

# 运行脚本
docker run --rm -v $(pwd):/app python-base:3.12 python test.py
```

### 场景 3：运行简单的 FastAPI 应用

```bash
# 创建简单的 FastAPI 应用
cat > simple_app.py << EOF
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from base image!"}

@app.get("/health")
def health():
    return {"status": "healthy"}
EOF

# 运行应用
docker run -d --name simple-app \
  -v $(pwd):/app \
  -p 8000:8000 \
  python-base:3.12 \
  uvicorn simple_app:app --host 0.0.0.0 --port 8000
# 测试
curl http://localhost:8000/health
```

### 场景 4：交互式开发

```bash
# 进入容器进行交互式开发
docker run -it --rm \
  -v $(pwd):/app \
  python-base:3.12 \
  /bin/bash

# 在容器内可以：
# - 运行 Python 代码
# - 安装额外的包（pip install package_name）
# - 测试代码
```

### 常用命令

```bash
# 查看基础镜像信息
docker images python-base:3.12
# 查看已安装的包
docker run --rm python-base:3.12 pip list
# 测试特定包
docker run --rm python-base:3.12 python -c "import fastapi; print('OK')"
# 查看镜像大小
docker images python-base:3.12
# 进入容器检查
docker run -it --rm python-base:3.12 /bin/bash


# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker
# 重启 Docker 服务
sudo systemctl restart docker
# 查看运行中的容器
docker ps

# 查看容器日志
docker logs -f ckb-qa-tool

# 进入容器
docker exec -it ckb-qa-tool bash

# 重新构建镜像
# 清理旧镜像
docker rmi ckb-qa-tool:latest
# 重新构建
docker build -t ckb-qa-tool:latest .
# 或使用 Docker Compose
docker-compose build --no-cache
```

## 端口映射说明

基础镜像暴露了以下端口，可以根据需要映射：

- `8000` - 常用 Web 服务
- `8010` - 常用 API 服务
- `8080` - 备用 Web 端口
- `8888` - Jupyter/开发端口
- `5000` - Flask 默认端口

示例：
```bash
# 映射到不同端口
docker run -p 9000:8000 python-base:3.12 uvicorn app:app --port 8000
# 外部访问：http://localhost:9000
```


## 错误排查

```bash
# 1. 端口已被占用  Error: bind: address already in use

# 查找占用端口的进程
netstat -tulpn | grep 8010
# 或
lsof -i :8010

# 停止占用端口的服务，或修改 docker-compose.yml 中的端口映射
# 例如改为 8011:8010
ports:
  - "8011:8010"


# 2. 权限问题  Permission denied
# 确保数据目录有正确的权限
chmod -R 755 data/
chmod -R 755 logs/

# 或者使用 root 用户运行
docker run --user root ...


# 3. 容器无法访问外部网络
# 检查网络配置
docker network ls

# 使用 host 网络模式（不推荐，但可以解决网络问题）
docker run --network host ...
```

## 数据持久化

数据目录已通过卷挂载到宿主机：
- `./data` → `/app/data`（输入输出文件）
- `./logs` → `/app/logs`（日志文件）

即使删除容器，数据也会保留在宿主机上。


## 更新代码后重启

```bash
# 方法 1：重新构建并启动
docker-compose up -d --build

# 方法 2：仅重启容器（代码已通过卷挂载）
docker-compose restart
```