#!/bin/bash

# Docker 快速启动脚本
# 使用方法: ./docker-run.sh [start|stop|restart|logs|build|shell]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker 服务未运行，请启动 Docker 服务"
        print_info "运行: sudo systemctl start docker"
        exit 1
    fi
}

# 检查 Docker Compose 是否安装
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_warn "Docker Compose 未安装，将使用 docker 命令代替"
        return 1
    fi
    return 0
}

# 使用 docker-compose 或 docker compose
compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        docker-compose "$@"
    elif docker compose version &> /dev/null; then
        docker compose "$@"
    else
        print_error "无法找到 docker-compose 或 docker compose 命令"
        exit 1
    fi
}

# 启动服务
start_service() {
    print_info "启动 CKB QA Tool 服务..."
    
    # 检查镜像是否存在
    if ! docker images | grep -q ckb-qa-tool; then
        print_warn "镜像不存在，开始构建..."
        build_image
    fi
    
    compose_cmd up -d
    
    print_info "等待服务启动..."
    sleep 3
    
    # 检查服务状态
    if compose_cmd ps | grep -q "Up"; then
        print_info "服务已启动！"
        print_info "API 地址: http://localhost:8010"
        print_info "健康检查: curl http://localhost:8010/health"
        print_info "查看日志: ./docker-run.sh logs"
    else
        print_error "服务启动失败，请查看日志: ./docker-run.sh logs"
        exit 1
    fi
}

# 停止服务
stop_service() {
    print_info "停止 CKB QA Tool 服务..."
    compose_cmd stop
    print_info "服务已停止"
}

# 重启服务
restart_service() {
    print_info "重启 CKB QA Tool 服务..."
    compose_cmd restart
    print_info "服务已重启"
}

# 查看日志
show_logs() {
    print_info "显示服务日志（按 Ctrl+C 退出）..."
    compose_cmd logs -f
}

# 构建镜像
build_image() {
    print_info "构建 Docker 镜像..."
    compose_cmd build --no-cache
    print_info "镜像构建完成"
}

# 进入容器
enter_shell() {
    print_info "进入容器 shell..."
    compose_cmd exec ckb-qa-tool bash
}

# 显示状态
show_status() {
    print_info "服务状态:"
    compose_cmd ps
    
    echo ""
    print_info "容器资源使用:"
    docker stats --no-stream ckb-qa-tool 2>/dev/null || print_warn "容器未运行"
}

# 清理
cleanup() {
    print_warn "这将删除容器和网络，但保留镜像和数据"
    read -p "确认继续? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        compose_cmd down
        print_info "清理完成"
    else
        print_info "已取消"
    fi
}

# 主函数
main() {
    check_docker
    check_docker_compose
    
    case "${1:-start}" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        logs)
            show_logs
            ;;
        build)
            build_image
            ;;
        shell)
            enter_shell
            ;;
        status)
            show_status
            ;;
        clean)
            cleanup
            ;;
        *)
            echo "使用方法: $0 [start|stop|restart|logs|build|shell|status|clean]"
            echo ""
            echo "命令说明:"
            echo "  start   - 启动服务（默认）"
            echo "  stop    - 停止服务"
            echo "  restart - 重启服务"
            echo "  logs    - 查看日志"
            echo "  build   - 构建镜像"
            echo "  shell   - 进入容器 shell"
            echo "  status  - 查看服务状态"
            echo "  clean   - 清理容器和网络"
            exit 1
            ;;
    esac
}

main "$@"




