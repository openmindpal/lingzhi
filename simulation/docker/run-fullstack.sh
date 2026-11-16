#!/bin/bash

#
# 灵智系统全栈在环验证——被验系统 = 灵智系统（引擎为灵智系统专有组件）
# 实证断言：
#
# 用法：bash simulation/docker/run-fullstack.sh
# 前置：docker 运行中；无凭据需求（mock LLM）
set -e
export MSYS2_ARG_CONV_EXCL="*"
cd "$(dirname "$0")/../.."

COMPOSE="docker compose -f e2e/docker-compose.e2e.yml -f simulation/docker/compose.fullstack.yaml -p mindpal-e2e"
API="http://127.0.0.1:17801"
TOKEN="admin@space_dev"

echo "=== 0. 快照用户模型绑定（down -v 清卷前——绑定跨运行保留，永不清掉）==="
BINDING_SNAPSHOT=".tmp/binding-snapshot.json"
$COMPOSE exec -T postgres psql -U mindpal -d mindpal -t -c "
  SELECT json_agg(row_to_json(t)) FROM (
    SELECT b.model_ref, b.provider, b.model, b.base_url, b.chat_completions_path,
           b.connector_instance_id, b.secret_id, b.secret_ids, b.status,
           ci.type_name, ci.egress_policy,
           (SELECT json_agg(sr.*) FROM secret_records sr WHERE sr.connector_instance_id = b.connector_instance_id) AS secrets
    FROM provider_bindings b
    JOIN connector_instances ci ON ci.id = b.connector_instance_id
    WHERE b.tenant_id = 'tenant_dev' AND b.model_ref != 'mock:scripted'
  ) t;" > "$BINDING_SNAPSHOT" 2>/dev/null || echo "[]" > "$BINDING_SNAPSHOT"
echo "  绑定快照: $(head -c 120 "$BINDING_SNAPSHOT")"

echo "=== 1. 起全栈（e2e 基座 + 仿真 override；mock-llm 带生命周期脚本重建 + api 代码更新重建）==="
$COMPOSE down -v --remove-orphans 2>&1 | tail -1
$COMPOSE build mock-llm api 2>&1 | tail -1
$COMPOSE up -d 2>&1 | tail -3

echo "=== 2. api 健康 ==="
for i in $(seq 1 60); do
  curl -fsS -m 5 $API/v1/health >/dev/null 2>&1 && { echo "api OK"; break; }
  [ "$i" = 60 ] && { echo "api 未就绪"; exit 1; }
  sleep 3
done

echo "=== 3. 种子 ==="
MSYS2_ARG_CONV_EXCL="*" $COMPOSE exec -T api node /e2e/seed.mjs | tail -1

echo "=== 3.5 恢复用户模型绑定（快照回放——密钥信封同 masterKey 可解密）+ 切路由 ==="
$COMPOSE cp "$BINDING_SNAPSHOT" api:/tmp/binding-snapshot.json >/dev/null 2>&1
$COMPOSE exec -T api node -e "
const { createRequire } = require('module');
const req = createRequire('/app/apps/api/dist/index.js');
const fs = require('fs');
(async () => {
  const snap = JSON.parse(fs.readFileSync('/tmp/binding-snapshot.json', 'utf8') || '[]');
  if (!snap.length) { console.log('无用户绑定快照——保持 mock 自举'); return; }
  const { loadConfig } = req('/app/apps/api/dist/config.js');
  const { createPool } = req('/app/apps/api/dist/db/pool.js');
  const cfg = loadConfig(process.env);
  const pool = createPool(cfg);
  for (const b of snap) {
    await pool.query(
      \`INSERT INTO connector_instances (id, tenant_id, scope_type, scope_id, name, type_name, status, egress_policy)
       VALUES (\$1,'tenant',\$1,'restored-'\$2,\$3,'enabled',\$4::jsonb)
       ON CONFLICT (id) DO UPDATE SET status='enabled', egress_policy=\$4::jsonb\`,
      [b.connector_instance_id, b.model_ref, b.type_name, JSON.stringify(b.egress_policy ?? {})]);
    if (b.secrets) for (const sr of b.secrets) {
      await pool.query(
        \`INSERT INTO secret_records (id, tenant_id, scope_type, scope_id, connector_instance_id, status, key_version, encrypted_payload)
         VALUES (\$1,\$2,'tenant',\$1,\$3,\$4,\$5,\$6::jsonb)
         ON CONFLICT (id) DO UPDATE SET encrypted_payload=\$6::jsonb, status=\$4\`,
        [sr.id, 'tenant_dev', b.connector_instance_id, sr.status, sr.key_version ?? 1, JSON.stringify(sr.encrypted_payload)]);
    }
    await pool.query(
      \`INSERT INTO provider_bindings (tenant_id, scope_type, scope_id, model_ref, provider, model, base_url, chat_completions_path, connector_instance_id, secret_id, secret_ids, status)
       VALUES (\$1,'tenant',\$1,\$2,\$3,\$4,\$5,\$6,\$7,\$8,\$9::jsonb,'enabled')
       ON CONFLICT (id) DO NOTHING\`,
      ['tenant_dev', b.model_ref, b.provider, b.model, b.base_url, b.chat_completions_path, b.connector_instance_id, b.secret_id, JSON.stringify(b.secret_ids ?? [])]);
    for (const purpose of ['agent.loop.decide','agent.loop.reflect','agent.loop.decompose']) {
      await pool.query(
        \`INSERT INTO routing_policies (tenant_id, purpose, primary_model_ref, fallback_model_refs, enabled)
         VALUES (\$1,\$2,\$3,'[]'::jsonb,true)
         ON CONFLICT (tenant_id, purpose) DO UPDATE SET primary_model_ref=\$3, enabled=true\`,
        ['tenant_dev', purpose, b.model_ref]);
    }
    console.log('restored:', b.model_ref);
  }
  await pool.end();
})();" 2>&1 | grep -E "restored|无用户" | head -3

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
DEV_RES=$(curl -fsS -m 10 -X POST $API/v1/devices -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d '{"deviceType":"robot","os":"container-sim","agentVersion":"sim-fullstack"}')
DEVICE_ID=$(printf '%s' "$DEV_RES" | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{try{console.log(JSON.parse(d).device.deviceId)}catch{console.log('')}})")
[ -n "$DEVICE_ID" ] || { echo "设备创建失败: $DEV_RES"; exit 1; }
PAIRING_CODE=$(curl -fsS -m 10 -X POST $API/v1/devices/$DEVICE_ID/pairing -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d '{}' | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{try{console.log(JSON.parse(d).pairingCode)}catch{console.log('')}})")
[ -n "$PAIRING_CODE" ] || { echo "配对码失败"; exit 1; }
$COMPOSE exec -T device-agent mindpal-device-agent pair --pairing-code "$PAIRING_CODE" 2>&1 | grep -aoE 'pairing successful' | tail -1

echo "=== 6. 本体包重签（dev 密钥——签名信任链）==="
rm -rf .tmp/golden-c2-package && mkdir -p .tmp/golden-c2-package
cp -r simulation/packages/sim-arm-6dof/. .tmp/golden-c2-package/
GOLDEN_KEY=".tmp/golden_dev_key.pem"
python apps/body-packages/tools/sign_package.py --generate "$GOLDEN_KEY" > /dev/null
C2_PUBKEY=$(python apps/body-packages/tools/sign_package.py .tmp/golden-c2-package "$GOLDEN_KEY" | sed -n 's/^PUBKEY_B64=//p')
[ -n "$C2_PUBKEY" ] || { echo "重签失败"; exit 1; }

echo "  docker cp 本体包+感知模型入容器（bind mount 在 Windows 有陈旧读缓存——字节精确投递）"
docker cp .tmp/golden-c2-package/. mindpal-e2e-device-agent-1:/pkg/arm/
$COMPOSE exec -T device-agent mkdir -p /build/models/grounding_dino_tiny
docker cp apps/device-agent-rs/models/yolox_nano.onnx mindpal-e2e-device-agent-1:/build/models/yolox_nano.onnx
docker cp apps/device-agent-rs/models/grounding_dino_tiny/tokenizer.json mindpal-e2e-device-agent-1:/build/models/grounding_dino_tiny/tokenizer.json
docker cp apps/device-agent-rs/models/grounding_dino_tiny/model_quantized.onnx mindpal-e2e-device-agent-1:/build/models/grounding_dino_tiny/model_quantized.onnx
docker cp apps/device-agent-rs/models/grounding_dino_tiny/config.json mindpal-e2e-device-agent-1:/build/models/grounding_dino_tiny/config.json

echo "=== 7. resident MuJoCo 身体就绪等待（compose 主进程守护循环自启）==="
for i in $(seq 1 20); do
  CAM=$($COMPOSE exec -T device-agent /opt/venv/bin/python -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8284/frame.jpg', timeout=3); print('OK')" 2>/dev/null | tr -d ' 
')
  [ "$CAM" = "OK" ] && { echo "  resident + camera 就绪"; break; }
  [ "$i" = 20 ] && { echo "  resident/camera 未就绪"; exit 1; }
  sleep 3
done

echo "=== 8. 引擎 Run（WS + 本体包装载）==="
$COMPOSE exec -d -e HARDWARE_BACKEND=auto -e MINDPAL_BODY_PACKAGE=/pkg/arm -e MINDPAL_BODY_PACKAGE_PUBKEYS="$C2_PUBKEY" -e MINDPAL_MODELS_DIR=/build/models -e MINDPAL_GDINO_DIR=/build/models/grounding_dino_tiny device-agent sh -c 'mindpal-device-agent run > /tmp/engine-run.log 2>&1'
for i in $(seq 1 100); do
  ACT=$($COMPOSE exec -T postgres psql -U mindpal -d mindpal -t -c "SELECT count(*) FROM tool_definitions d WHERE d.name='device.actuator.gesture';" 2>/dev/null | tr -d ' \n')
  [ "${ACT:-0}" -ge 1 ] 2>/dev/null && { echo "actuator 能力落库"; break; }
  [ "$i" = 100 ] && { echo "能力未落库（心跳链断）"; exit 1; }
  sleep 1
done
echo "  重启 api 清工具目录缓存（能力落库后——运行期工具目录冷启动期会拒派发）"
$COMPOSE restart api mock-llm >/dev/null 2>&1
for i in $(seq 1 40); do
  curl -fsS -m 5 $API/v1/health >/dev/null 2>&1 && { echo "api 重启 OK"; break; }
  [ "$i" = 40 ] && { echo "api 重启未就绪"; exit 1; }
  sleep 3
done
sleep 3

echo "=== 9. A1 认知循环在环（goal → 云端认知循环 → agent_done）==="
node simulation/docker/a1-cognitive.mjs "$API"

echo "=== 10. A2 具身派发闭环（greet_wave → 引擎 → MuJoCo resident）==="
C2=$(curl -fsS -m 15 -X POST $API/v1/device-executions -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d "{\"deviceId\":\"$DEVICE_ID\",\"toolRef\":\"device.actuator.gesture\",\"input\":{\"intent\":\"greet_wave\",\"max_duration_ms\":2500},\"requireUserPresence\":false}")
EXEC_ID=$(printf '%s' "$C2" | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{try{console.log(JSON.parse(d).execution.deviceExecutionId)}catch{console.log('')}})")
[ -n "$EXEC_ID" ] || { echo "FAIL[A2] 派发失败: $C2"; exit 1; }
for i in $(seq 1 60); do
  ST=$(curl -fsS -m 10 $API/v1/device-executions/$EXEC_ID -H "authorization: Bearer $TOKEN" | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{try{console.log(JSON.parse(d).execution.status)}catch{console.log('')}})")
  [ "$ST" = "succeeded" -o "$ST" = "failed" ] && break
  sleep 1
done
[ "$ST" = "succeeded" ] || { echo "FAIL[A2] 终态=$ST"; exit 1; }
echo "A2 PASS 具身闭环：云端派发 → 引擎策略行推理 → DDS → MuJoCo resident → 成功回执"

echo "=== 11. A3 生命周期断言（ask_user 挂起→续跑→checkpoint；迭代硬帽）==="
node simulation/docker/a3-lifecycle.mjs "$API"

echo "=== 12. A4 厨房编排（goal → 三词顺序派发 → 策略闭环 ×3 → done）==="
node simulation/docker/a4-kitchen.mjs "$API" "$DEVICE_ID"

echo "=== 13. A4-DB 厨房三词派发落库核验（device_executions 为准）==="
KC=$($COMPOSE exec -T postgres psql -U mindpal -d mindpal -t -c "SELECT count(DISTINCT (de.input_json->>'goal_x') || ',' || (de.input_json->>'goal_y') || ',' || (de.input_json->>'goal_z')) FROM device_executions de WHERE de.tool_ref='device.actuator.gesture' AND de.status='succeeded' AND de.input_json->>'intent'='reach_goal' AND de.input_json->>'goal_y' IN ('0.12','-0.05');" 2>/dev/null | tr -d ' 
')
[ "${KC:-0}" -ge 2 ] 2>/dev/null && echo "A4-DB PASS 厨房 goal 参数化派发落库 succeeded（$KC 个不同目标点）" || { echo "FAIL[A4-DB] 厨房 goal 点落库 $KC"; exit 1; }

echo "=== 14. A5 工厂编排（goal → 三词顺序派发 → 策略闭环 ×3 → done + DB 核验）==="
node simulation/docker/a5-factory.mjs "$API" "$DEVICE_ID"

echo "=== 15. A6 交通指挥（goal → 三方位派发 → 策略闭环 ×3 → done + DB 核验）==="
node simulation/docker/a6-traffic.mjs "$API" "$DEVICE_ID"

echo "FULLSTACK_SIM_PASS"
