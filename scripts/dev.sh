#!/usr/bin/env bash
# 本项目 CUDA 调试运行环境管理脚本
# 用法: ./scripts/dev.sh {up|down|shell|build|run|stop|status|logs|restart}
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="docker-compose.dev.yml"
SERVICE="cuda-dev"
CONTAINER="embedding-cuda-dev"

# 可通过环境变量覆盖；PORT 不同即可在同一容器内并行多个模型实例
BUILD_DIR="${BUILD_DIR:-build-cuda}"
MODEL_PATH="${MODEL_PATH:-/Models/models-hf/sentence_models/bge-large-zh-v1.5}"
MODEL_NAME="${MODEL_NAME:-bge-large-zh-v1.5}"
PORT="${PORT:-8080}"
JOBS="${JOBS:-$(nproc)}"
# 日志与 PID 按端口区分，支持 8080/8081 多实例共存
SERVER_LOG="/tmp/embedding-server-${PORT}.log"
SERVER_PID="/tmp/embedding-server-${PORT}.pid"

dc() { docker compose -f "$COMPOSE_FILE" "$@"; }

in_container() { dc exec -T "$SERVICE" bash -lc "$1"; }

wait_ready() {
  for _ in $(seq 1 40); do
    if docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
      echo "容器已就绪: $CONTAINER"
      return 0
    fi
    sleep 0.5
  done
  echo "容器启动超时，请执行 docker compose -f $COMPOSE_FILE logs" >&2
  return 1
}

case "${1:-status}" in
  up)
    dc up -d
    wait_ready
    in_container 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'
    echo "进入调试 shell: ./scripts/dev.sh shell"
    ;;

  down)
    ./scripts/dev.sh stop || true
    dc down
    ;;

  shell)
    dc exec "$SERVICE" bash
    ;;

  build)
    echo ">>> 配置 CMake (USE_CUDA=ON, ENABLE_SERVER=ON) -> $BUILD_DIR"
    in_container "cmake -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DENABLE_SERVER=ON"
    echo ">>> 编译 (jobs=$JOBS)"
    in_container "cmake --build $BUILD_DIR -j$JOBS"
    echo ">>> 产物:"
    in_container "ls -lh $BUILD_DIR/server $BUILD_DIR/app $BUILD_DIR/cli $BUILD_DIR/benchmark_cpp 2>/dev/null"
    ;;

  run)
    echo ">>> 启动服务: $MODEL_PATH (port $PORT)"
    in_container "test -x $BUILD_DIR/server" || { echo "未找到 $BUILD_DIR/server，先执行 ./scripts/dev.sh build" >&2; exit 1; }
    in_container "test -d $MODEL_PATH" || { echo "模型目录不存在: $MODEL_PATH" >&2; exit 1; }
    # 先清理旧进程，避免端口占用
    in_container "if [ -f $SERVER_PID ]; then kill \$(cat $SERVER_PID) 2>/dev/null || true; rm -f $SERVER_PID; fi"
    in_container "nohup ./$BUILD_DIR/server --model_path $MODEL_PATH --serve_model_name $MODEL_NAME --port $PORT > $SERVER_LOG 2>&1 & echo \$! > $SERVER_PID"
    for _ in $(seq 1 120); do
      if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
         || grep -q "Starting server on port" <(in_container "cat $SERVER_LOG" 2>/dev/null); then
        break
      fi
      sleep 1
    done
    in_container "cat $SERVER_LOG"
    echo ">>> 服务已监听 http://127.0.0.1:$PORT"
    ;;

  stop)
    if docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
      # 只停本端口实例，不影响同容器内的其它模型（其它端口由其自身 PID 文件管理）
      in_container "if [ -f $SERVER_PID ]; then kill \$(cat $SERVER_PID) 2>/dev/null || true; rm -f $SERVER_PID; echo 服务已停止（端口 $PORT）; else echo 未发现端口 $PORT 的服务; fi"
    fi
    ;;

  logs)
    in_container "tail -n ${LINES:-100} -f $SERVER_LOG"
    ;;

  status)
    docker ps --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    if docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
      echo "--- GPU ---"
      in_container 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
      echo "--- 服务进程 ---"
      in_container "if [ -f $SERVER_PID ] && kill -0 \$(cat $SERVER_PID) 2>/dev/null; then echo \"PID \$(cat $SERVER_PID) 运行中\"; else echo 未运行; fi"
      echo "--- 接口自测 ---"
      curl -s -m 5 -X POST "http://127.0.0.1:$PORT/embed" \
        -H 'Content-Type: application/json' \
        -d '{"text":"你好，世界"}' | head -c 200 || echo "(无响应)"
      echo
    fi
    ;;

  restart)
    ./scripts/dev.sh stop
    ./scripts/dev.sh run
    ;;

  *)
    echo "用法: $0 {up|down|shell|build|run|stop|restart|status|logs}" >&2
    exit 1
    ;;
esac
