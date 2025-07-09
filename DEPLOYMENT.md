# GenBank Gene Extractor 部署指南

## 服务器部署

本项目提供了一个便捷的启动脚本 `run.py`，可以直接在服务器上运行，无需手动执行 `streamlit run main.py` 命令。

### 快速启动

```bash
# 直接启动（推荐）
python run.py

# 或者使用 python3
python3 run.py
```

### 启动选项

```bash
# 指定端口
python run.py --port 8080

# 仅本地访问
python run.py --host 127.0.0.1

# 启动后自动打开浏览器（本地环境）
python run.py --browser

# 不自动安装依赖包
python run.py --no-install

# 组合使用
python run.py --port 8080 --host 0.0.0.0
```

### 功能特性

✅ **自动依赖检查** - 自动检测并安装缺少的依赖包  
✅ **智能端口选择** - 自动查找可用端口  
✅ **环境配置** - 自动设置所需的环境变量  
✅ **错误处理** - 完善的错误处理和用户提示  
✅ **实时日志** - 实时显示应用运行状态  
✅ **优雅停止** - 支持 Ctrl+C 优雅停止应用  

### 服务器部署步骤

1. **上传项目文件**
   ```bash
   # 将整个项目文件夹上传到服务器
   scp -r Gene_Pipeline/ user@server:/path/to/deployment/
   ```

2. **进入项目目录**
   ```bash
   cd /path/to/deployment/Gene_Pipeline
   ```

3. **启动应用**
   ```bash
   # 默认启动（端口8501，允许外部访问）
   python run.py
   
   # 或指定端口
   python run.py --port 8080
   ```

4. **访问应用**
   - 本地访问: `http://localhost:8501`
   - 外部访问: `http://服务器IP:8501`

### 后台运行

如果需要在后台运行应用，可以使用以下方法：

```bash
# 使用 nohup 后台运行
nohup python run.py --port 8501 > app.log 2>&1 &

# 使用 screen 会话
screen -S genbank_app
python run.py
# 按 Ctrl+A 然后 D 来分离会话

# 重新连接到 screen 会话
screen -r genbank_app
```

### 系统服务配置

创建 systemd 服务文件 `/etc/systemd/system/genbank-extractor.service`：

```ini
[Unit]
Description=GenBank Gene Extractor
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Gene_Pipeline
ExecStart=/usr/bin/python3 run.py --port 8501
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable genbank-extractor
sudo systemctl start genbank-extractor
sudo systemctl status genbank-extractor
```

### 防火墙配置

如果服务器有防火墙，需要开放相应端口：

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 8501

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### 故障排除

1. **端口被占用**
   - 启动器会自动查找可用端口
   - 或手动指定其他端口: `python run.py --port 8080`

2. **依赖包缺失**
   - 启动器会自动安装依赖
   - 如果自动安装失败，手动安装: `pip install -r requirements.txt`

3. **权限问题**
   - 确保用户有读写项目目录的权限
   - 确保用户有安装Python包的权限

4. **网络访问问题**
   - 检查防火墙设置
   - 确保使用 `--host 0.0.0.0` 允许外部访问

### 日志查看

应用运行时会显示实时日志，包括：
- 启动状态
- 依赖检查结果
- 访问地址
- 错误信息

### 性能优化

对于生产环境，建议：

1. **使用反向代理**
   ```nginx
   # Nginx 配置示例
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

2. **资源限制**
   - 根据服务器配置调整内存和CPU使用
   - 监控应用资源使用情况

3. **备份策略**
   - 定期备份生成的结果文件
   - 备份配置文件

### 支持的操作系统

- ✅ Linux (Ubuntu, CentOS, Debian, etc.)
- ✅ macOS
- ✅ Windows (使用 `python run.py`)

### 最低系统要求

- Python 3.8+
- 2GB RAM (推荐 4GB+)
- 1GB 磁盘空间
- 网络连接 (用于下载GenBank数据) 