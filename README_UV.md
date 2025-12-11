# 加速安装指南

## 问题说明

`uv sync` 在下载 Python 解释器时可能遇到网络超时问题，特别是在国内网络环境下。

## 解决方案

### 方案 1：使用环境变量增加超时时间（推荐）

在执行 `uv sync` 前设置环境变量：

```bash
# 增加超时时间到 10 分钟
export UV_HTTP_TIMEOUT=600
export UV_NETWORK_RETRIES=5

# 然后执行同步
cd /root/zhongli2/iflytek_ckb_tool
uv sync
```

### 方案 2：使用已配置的国内镜像

项目已配置清华 PyPI 镜像源（在 `pyproject.toml` 中），全局配置也已更新（`~/.config/uv/uv.toml`）。

直接重试：

```bash
cd /root/zhongli2/iflytek_ckb_tool
uv sync
```

### 方案 3：手动安装 Python 3.12+ 后使用系统 Python

如果网络问题持续，可以手动安装 Python 3.12+：

#### 3.1 使用 pyenv 安装（推荐）

```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 添加到 PATH
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# 安装 Python 3.12
pyenv install 3.12.7

# 在项目目录设置 Python 版本
cd /root/zhongli2/iflytek_ckb_tool
pyenv local 3.12.7

# 使用系统 Python
uv sync --python $(which python3.12)
```

#### 3.2 从源码编译安装

```bash
# 安装编译依赖
yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel readline-devel sqlite-devel

# 下载 Python 3.12.7
cd /tmp
wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
tar -xzf Python-3.12.7.tgz
cd Python-3.12.7

# 编译安装
./configure --prefix=/usr/local/python3.12 --enable-optimizations
make -j$(nproc)
make altinstall

# 创建软链接
ln -s /usr/local/python3.12/bin/python3.12 /usr/local/bin/python3.12

# 使用系统 Python
cd /root/zhongli2/iflytek_ckb_tool
uv sync --python /usr/local/bin/python3.12
```

### 方案 4：使用代理（如果有）

如果有可用的代理，可以设置：

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
uv sync
```

### 方案 5：分步下载和重试

如果下载中断，可以清理缓存后重试：

```bash
# 清理 Python 缓存
uv cache clean python

# 重试同步（会自动续传）
uv sync
```

## 验证安装

安装完成后，验证环境：

```bash
# 检查 Python 版本
uv run python --version

# 检查依赖是否安装成功
uv run python -c "import fastapi; print('FastAPI installed')"
```

## 常见问题

### Q: 仍然超时怎么办？

A: 尝试以下方法：
1. 在网络较好的时间段重试
2. 使用方案 3 手动安装 Python
3. 检查防火墙设置

### Q: 如何查看 uv 的下载进度？

A: 使用 `--verbose` 参数：
```bash
uv sync --verbose
```

### Q: 可以离线安装吗？

A: 可以，但需要先在有网络的机器上下载好 Python 解释器和依赖包，然后传输到目标机器。

## 当前配置状态

- ✅ PyPI 镜像：已配置清华镜像（`pyproject.toml`）
- ✅ 全局配置：已更新 `~/.config/uv/uv.toml`
- ⚠️ Python 解释器：需要从官方源下载（无国内镜像）

