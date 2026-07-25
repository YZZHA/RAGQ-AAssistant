#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FDI RAG QA Assistant — 一键部署脚本
# 适用: Ubuntu 22.04 LTS
# 用法: chmod +x deploy.sh && sudo ./deploy.sh
# ============================================================

APP_DIR="/home/ubuntu/rag-qa-assistant"
APP_USER="ubuntu"
VENV_DIR="$APP_DIR/venv"
GIT_REPO="https://github.com/YOUR_USERNAME/rag-qa-assistant.git"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# --------------------------------------------------
# 1. 检查 root
# --------------------------------------------------
if [ "$EUID" -ne 0 ]; then
    err "请以 root 身份运行: sudo ./deploy.sh"
fi

log "开始部署 FDI RAG QA Assistant..."

# --------------------------------------------------
# 2. 系统依赖
# --------------------------------------------------
log "安装系统依赖..."
apt update -qq
apt install -y -qq python3.11 python3.11-venv python3.11-dev nginx git curl 2>/dev/null

# --------------------------------------------------
# 3. 克隆/更新项目
# --------------------------------------------------
if [ -d "$APP_DIR" ]; then
    warn "项目目录已存在，执行 git pull..."
    cd "$APP_DIR"
    git pull
else
    log "克隆项目..."
    git clone "$GIT_REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

# --------------------------------------------------
# 4. Python 虚拟环境 + 依赖
# --------------------------------------------------
log "创建虚拟环境..."
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

log "安装 Python 依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>/dev/null || pip install -q uvicorn fastapi pydantic-settings openai sentence-transformers jieba fakeredis pypdf python-docx

deactivate

# --------------------------------------------------
# 5. .env 文件
# --------------------------------------------------
if [ ! -f "$APP_DIR/.env" ]; then
    warn ".env 文件不存在，从 .env.example 复制，请手动填入 QWEN_API_KEY"
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    sed -i 's/EMBEDDING_MODEL=text2vec-large-chinese/EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2/' "$APP_DIR/.env"
fi

# --------------------------------------------------
# 6. 部署配置文件
# --------------------------------------------------
log "配置 Nginx..."
cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/rag-qa
ln -sf /etc/nginx/sites-available/rag-qa /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

log "配置 Systemd 服务..."
cp "$APP_DIR/deploy/rag-qa.service" /etc/systemd/system/rag-qa.service
systemctl daemon-reload
systemctl enable rag-qa

# --------------------------------------------------
# 7. 目录权限
# --------------------------------------------------
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# --------------------------------------------------
# 8. 创建 Swap（2GB，防止 OOM）
# --------------------------------------------------
if ! swapon --show | grep -q /swapfile; then
    log "创建 2GB Swap..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --------------------------------------------------
# 9. 启动服务
# --------------------------------------------------
log "启动服务..."
systemctl start rag-qa

# --------------------------------------------------
# 10. 验证
# --------------------------------------------------
sleep 3
if systemctl is-active --quiet rag-qa; then
    log "服务运行中"
else
    err "服务启动失败，查看日志: journalctl -u rag-qa -n 50"
fi

if curl -sf http://127.0.0.1/health > /dev/null 2>&1; then
    log "健康检查通过 ✅"
    IP=$(curl -sf http://checkip.amazonaws.com 2>/dev/null || echo "获取公网 IP 失败")
    echo ""
    echo "============================================"
    echo -e "  ${GREEN}部署完成!${NC}"
    echo "  访问地址: http://$IP"
    echo "  健康检查: http://$IP/health"
    echo "  查看日志: journalctl -u rag-qa -f"
    echo "  重启服务: sudo systemctl restart rag-qa"
    echo "============================================"
else
    warn "健康检查失败，请手动排查: curl http://127.0.0.1/health"
fi
