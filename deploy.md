# Detextify 部署文档

## 环境要求

- Python 3.10
- 操作系统：Windows / Linux / macOS

## 安装步骤

### 1. 创建虚拟环境

**Windows (PowerShell):**
```powershell
python -m venv detextify_env
.\detextify_env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv detextify_env
source detextify_env/bin/activate
```

### 2. 升级基础工具

```bash
pip install --upgrade setuptools pip wheel
```

### 3. 安装依赖包

使用清华镜像源加速安装：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 4. 安装最新版 diffusers

```bash
pip install -U git+https://github.com/huggingface/diffusers
```

## 验证安装

运行以下命令验证安装是否成功：

```bash
python -c "import diffusers; print(diffusers.__version__)"
```

## 注意事项

1. 确保在安装依赖前已激活虚拟环境
2. 如果在国内网络环境，建议使用清华镜像源加速下载
3. Windows 用户如遇到执行策略限制，可运行：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
4. diffusers 库需要从 GitHub 安装最新版本以获取最新功能