# deploy/ — 生产部署配置

一键部署脚本及 Nginx + Systemd 配置文件。

## 文件清单

| 文件 | 说明 | 使用方式 |
|------|------|----------|
| `deploy.sh` | 一键部署脚本 | `sudo ./deploy.sh`（需先填入 GIT_REPO） |
| `nginx.conf` | Nginx 反向代理 + SSE 支持 | 由 deploy.sh 自动部署到 `/etc/nginx/sites-available/` |
| `rag-qa.service` | Systemd 服务守护 + 自动重启 | 由 deploy.sh 自动部署到 `/etc/systemd/system/` |

## 部署流程

```bash
# 1. 修改 deploy.sh 顶部的 GIT_REPO 为你的仓库地址
# 2. 上传 deploy/ 目录到 VPS
# 3. 执行
sudo chmod +x deploy.sh
sudo ./deploy.sh
# 4. 部署完成后，编辑 .env 填入 QWEN_API_KEY
sudo vi /home/ubuntu/rag-qa-assistant/.env
# 5. 重启服务
sudo systemctl restart rag-qa
```

## 注意事项

- 首次部署需要手动在 `.env` 填入 `QWEN_API_KEY`
- 脚本会自动创建 2GB Swap 防止 OOM
- Nginx 配置中 `/api/chat` 端点禁用了缓冲以支持 SSE
- Systemd 服务设置了 `MemoryMax=2G` 限制
