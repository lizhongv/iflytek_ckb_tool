
## 配置环境
```bash
pip install uv
uv sync

# source venv/bin/activate
# .venv\Scripts\activate
# deactivate
```

## 启动 APP

```bash
# uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level info
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info

# gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8010
```

###  开发环境
```bash
gunicorn app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8010 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  --reload
```

参数说明：
- `timeout 120`：worker 超时时间（秒）
- `access-logfile -`：访问日志输出到标准输出
- `error-logfile -`：错误日志输出到标准错误
- `log-level info`：日志级别
- `reload`：开发模式，代码变更自动重启（生产环境不建议使用）

###  生产环境
```bash
gunicorn app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8010 \
  --timeout 300 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --log-level info \
  --daemon
```
参数说明：
- `daemon`：后台运行
- 日志写入文件而非控制台

## 验证运行
```bash
# windows prot
netstat -ano | findstr :8010
# kill process
taskkill /PID 12342 /F

# 查找进程
ps aux | grep gunicorn
# 停止（替换 PID）
kill -9 <PID>
# 或使用 pkill
pkill -f gunicorn


# 检查端口是否监听
netstat -an | grep 8010
# 或
lsof -i :8010
# 测试 API
curl http://localhost:8010/health
```


