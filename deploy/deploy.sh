#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# FDI RAG QA Assistant — Docker 一键部署脚本
# 适用: Ubuntu 22.04 LTS / Debian 12+
# 用法: chmod +x deploy.sh && sudo ./deploy.sh
# ============================================================

APP_DIR="/home/ubuntu/rag-qa-assistant"
GIT_REPO="https://github.com/YZZHA/RAGQ-AAssistant.git"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

if [ "$EUID" -ne 0 ]; then
    err "请以 root 身份运行: sudo ./deploy.sh"
fi

log "开始部署 FDI RAG QA Assistant (Docker)..."

# --------------------------------------------------
# 1. 安装 Docker
# --------------------------------------------------
log "安装 Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
else
    log "Docker 已安装"
fi

if ! getent group docker | grep -q "$SUDO_USER"; then
    usermod -aG docker "$SUDO_USER"
    warn "已将 $SUDO_USER 添加到 docker 组，请重新登录后生效"
fi

# --------------------------------------------------
# 2. 克隆/更新项目
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
# 3. .env 文件
# --------------------------------------------------
if [ ! -f "$APP_DIR/.env" ]; then
    warn ".env 文件不存在，从 .env.example 复制，请手动填入 QWEN_API_KEY"
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    sed -i 's/EMBEDDING_MODEL=text2vec-large-chinese/EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2/' "$APP_DIR/.env"
fi

# --------------------------------------------------
# 4. 创建 Swap（2GB，防止 OOM）
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
# 5. 启动容器
# --------------------------------------------------
log "启动 Docker 容器..."
cd "$APP_DIR"
docker compose up -d

# --------------------------------------------------
# 6. 验证
# --------------------------------------------------
log "等待服务启动..."
sleep 10

if docker compose ps | grep -q "rag-qa-app.*running"; then
    log "容器运行中"
else
    err "容器启动失败，查看日志: docker compose logs -f"
fi

if curl -sf http://127.0.0.1/health > /dev/null 2>&1; then
    log "健康检查通过 ✅"
    IP=$(curl -sf http://checkip.amazonaws.com 2>/dev/null || hostname -I | awk '{print $1}')
    echo ""
    echo "============================================"
    echo -e "  ${GREEN}Docker 部署完成!${NC}"
    echo "  访问地址: http://$IP"
    echo "  健康检查: http://$IP/health"
    echo "  查看日志: docker compose logs -f"
    echo "  重启服务: docker compose restart"
    echo "  停止服务: docker compose down"
    echo ""
    echo "  ${YELLOW}提示:${NC} 首次启动需要下载模型，约需 5-10 分钟"
    echo "       可通过 docker compose logs -f 查看进度"
    echo "============================================"
else
    warn "健康检查失败，容器可能还在启动中..."
    warn "请等待几分钟后再试: curl http://127.0.0.1/health"
    warn "或查看日志: docker compose logs -f"
fi
