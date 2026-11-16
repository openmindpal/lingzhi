#!/bin/bash
# run-robo.sh——感知绑定在环矩阵验证（RoboDojo 型任务集·几何绑定维度）。
# 复用 fullstack 线（sim-verifier 镜像 = 引擎+python桥+osmesa），叠加 compose.robo.yaml：
#   场景 = robodojo_table（tri 本体 + 桌 + 目标盒 + 深度相机）
#   包   = robodojo-arm（tri-family 派生，http_depth 声明——gen_robo_package.py 生成重签）
# 链路 = MuJoCo 真渲染深度 → 引擎 http_depth 后端 → 抓取规划 → 机械判据（GT 真值路由）。
# 前置：镜像已重建（device-agent-rs:safety + mindpal-sim-verifier——含 http_depth/桥深度路）。
# 用法：bash simulation/docker/run-robo.sh        # 15 回合矩阵 → reports/robo_binding_*.json
set -e
export MSYS2_ARG_CONV_EXCL="*"
cd "$(dirname "$0")/../.."

COMPOSE="docker compose -f e2e/docker-compose.e2e.yml -f simulation/docker/compose.fullstack.yaml -f simulation/docker/compose.robo.yaml -p mindpal-e2e"
API="http://127.0.0.1:17801"
TOKEN="admin@space_dev"

echo "=== 1. 起全栈（robo override：robodojo 场景/包 + 端口旁路）==="
$COMPOSE down -v --remove-orphans 2>&1 | tail -1
$COMPOSE up -d 2>&1 | tail -3

echo "=== 2. api 健康 ==="
for i in $(seq 1 60); do
  curl -fsS -m 5 $API/v1/health >/dev/null 2>&1 && { echo "api OK"; break; }
  [ "$i" = 60 ] && { echo "api 未就绪"; exit 1; }
  sleep 3
done

echo "=== 3. 种子 ==="
MSYS2_ARG_CONV_EXCL="*" $COMPOSE exec -T api node /e2e/seed.mjs | tail -1

echo "=== 4. 工具热扫描 ==="
for i in $(seq 1 40); do
  CNT=$($COMPOSE exec -T postgres psql -U mindpal -d mindpal -t -c "SELECT count(*) FROM tool_definitions WHERE tenant_id='tenant_dev';" 2>/dev/null | tr -d ' \n')
  [ "${CNT:-0}" -ge 50 ] 2>/dev/null && { echo "tool_definitions=$CNT"; break; }
  [ "$i" = 40 ] && { echo "工具未注册"; exit 1; }
  sleep 5
done
$COMPOSE restart api mock-llm >/dev/null 2>&1
sleep 8

echo "=== 5. 引擎配对（真 Rust device-agent）==="
DEV_RES=$(curl -fsS -m 10 -X POST $API/v1/devices -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d '{"deviceType":"robot","os":"container-sim","agentVersion":"sim-robo"}')
DEVICE_ID=$(printf '%s' "$DEV_RES" | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{try{console.log(JSON.parse(d).device.deviceId)}catch{console.log('')}})")
[ -n "$DEVICE_ID" ] || { echo "设备创建失败: $DEV_RES"; exit 1; }
PAIRING_CODE=$(curl -fsS -m 10 -X POST $API/v1/devices/$DEVICE_ID/pairing -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d '{}' | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{try{console.log(JSON.parse(d).pairingCode)}catch{console.log('')}})")
[ -n "$PAIRING_CODE" ] || { echo "配对码失败"; exit 1; }
$COMPOSE exec -T device-agent mindpal-device-agent pair --pairing-code "$PAIRING_CODE" 2>&1 | grep -aoE 'pairing successful' | tail -1

echo "=== 6. robodojo 包生成+重签（http_depth 声明）+ 入容器 ==="
python simulation/tools/gen_robo_package.py
C2_PUBKEY=$(python apps/body-packages/tools/sign_package.py .tmp/robo-package .tmp/robo_dev_key.pem | sed -n 's/^PUBKEY_B64=//p')
[ -n "$C2_PUBKEY" ] || { echo "重签失败"; exit 1; }
$COMPOSE exec -T device-agent mkdir -p /pkg/robodojo
$COMPOSE cp .tmp/robo-package/. device-agent:/pkg/robodojo/

echo "=== 7. resident MuJoCo 身体就绪等待（深度供数端在场 = 场景/包对齐）==="
for i in $(seq 1 20); do
  D=$($COMPOSE exec -T device-agent /opt/venv/bin/python -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:8284/depth.raw', timeout=3)
    print('OK' if r.status == 200 and int(r.headers['Content-Length']) == 320*240*2 else 'BAD')
except Exception as e:
    print('NO')" 2>/dev/null | tr -d ' \r\n')
  [ "$D" = "OK" ] && { echo "  resident + 深度供数就绪（320×240 Z16）"; break; }
  [ "$i" = 20 ] && { echo "  resident/深度供数未就绪"; exit 1; }
  sleep 3
done

echo "=== 8. 引擎 Run（WS + robodojo 本体包装载——http_depth 后端接线）==="
$COMPOSE exec -d -e HARDWARE_BACKEND=auto -e MINDPAL_BODY_PACKAGE=/pkg/robodojo -e MINDPAL_BODY_PACKAGE_PUBKEYS="$C2_PUBKEY" device-agent sh -c 'mindpal-device-agent run > /tmp/engine-run.log 2>&1'
for i in $(seq 1 100); do
  ACT=$($COMPOSE exec -T postgres psql -U mindpal -d mindpal -t -c "SELECT count(*) FROM tool_definitions d WHERE d.name='device.grasp.plan';" 2>/dev/null | tr -d ' \n')
  [ "${ACT:-0}" -ge 1 ] 2>/dev/null && { echo "抓取规划能力落库"; break; }
  [ "$i" = 100 ] && { echo "能力未落库（心跳链断）"; exit 1; }
  sleep 1
done
echo "  重启 api 清工具目录缓存"
$COMPOSE restart api mock-llm >/dev/null 2>&1
for i in $(seq 1 40); do
  curl -fsS -m 5 $API/v1/health >/dev/null 2>&1 && { echo "api 重启 OK"; break; }
  [ "$i" = 40 ] && { echo "api 重启未就绪"; exit 1; }
  sleep 3
done
sleep 3

echo "=== 9. 感知绑定矩阵（15 回合：L1 基线 → L2 位置/尺寸/干扰 → L3 组合）==="
node simulation/docker/robo-binding.mjs "$API" "$DEVICE_ID"
