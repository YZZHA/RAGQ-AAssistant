# deploy/ — 生产部署配置

## 方式一：Docker 一键部署（推荐）

### 部署步骤

```bash
# 第一步：SSH 登录服务器
ssh ubuntu@你的服务器IP
# 输入密码 或 使用密钥认证

# 第二步：安装 Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit
# 然后重新连接
ssh ubuntu@你的服务器IP

# 验证安装
docker --version
docker compose version

# 第三步：Git Clone 项目
cd ~
git clone https://github.com/YZZHA/RAGQ-AAssistant.git
cd RAGQ-AAssistant

# 第四步：启动
docker compose up -d
```

首次执行会自动完成：
- 拉取 redis:7-alpine 镜像（~30MB）
- 根据 Dockerfile 构建 app 镜像（~5 分钟，含模型下载）
- 启动两个容器：rag-qa-app + rag-qa-redis

### 验证

```bash
# 查看运行状态
docker compose ps

# 查看日志
docker compose logs -f

# 测试健康检查
curl http://localhost/health

# 测试 API
curl -N http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"跨境投资登记系统有什么功能？"}'
```

### 管理命令

```bash
# 停止服务
docker compose down

# 重启服务
docker compose restart

# 更新代码
git pull && docker compose up -d --build

# 查看资源占用
docker stats
```

## 方式二：源码部署（Nginx + Systemd）

适用于需要自定义配置的场景。

### 文件清单

| 文件 | 说明 | 使用方式 |
|------|------|----------|
| `deploy.sh` | 一键部署脚本（源码方式） | `sudo ./deploy.sh` |
| `nginx.conf` | Nginx 反向代理 + SSE 支持 | 由 deploy.sh 自动部署 |
| `rag-qa.service` | Systemd 服务守护 + 自动重启 | 由 deploy.sh 自动部署 |

## 注意事项

- 首次部署需要在 `.env` 中填入 `QWEN_API_KEY`
- 脚本会自动创建 2GB Swap 防止 OOM
- 首次启动需要下载模型，约需 5-10 分钟
- 建议服务器配置：≥2GB 内存，≥2CPU 核
