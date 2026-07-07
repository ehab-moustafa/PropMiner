#!/usr/bin/env bash
# Start/restart prop-miner-official on SaladCloud and verify mining logs via SSH.
# Requires: curl, jq, SALAD_API_KEY in env.
set -euo pipefail

ORG="${SALAD_ORG:-appy}"
PROJECT="${SALAD_PROJECT:-default}"
CONTAINER="${SALAD_CONTAINER:-prop-miner-official}"
API_BASE="${SALAD_API_BASE:-https://api.salad.com/api/public}"
WATCH_SECS="${WATCH_SECS:-180}"

if [[ -z "${SALAD_API_KEY:-}" ]]; then
  echo "ERROR: export SALAD_API_KEY=..." >&2
  exit 1
fi

auth=(-H "Salad-Api-Key: ${SALAD_API_KEY}" -H "Accept: application/json")
cg_url="${API_BASE}/organizations/${ORG}/projects/${PROJECT}/containers/${CONTAINER}"
instances_url="${cg_url}/instances"

status="$(curl -sf "${auth[@]}" "${cg_url}" | jq -r '.current_state.status // empty')"
echo "==> Container group status: ${status}"

if [[ "${status}" == "stopped" ]]; then
  echo "==> POST start (portal Start button equivalent)"
  curl -sf -X POST "${auth[@]}" "${cg_url}/start" -o /dev/null -w "HTTP %{http_code}\n"
elif [[ "${status}" == "deploying" || "${status}" == "running" ]]; then
  instance_id="$(curl -sf "${auth[@]}" "${instances_url}" | jq -r '.instances[0].id // empty')"
  if [[ -n "${instance_id}" ]]; then
    echo "==> POST restart instance ${instance_id}"
    curl -sf -X POST "${auth[@]}" "${instances_url}/${instance_id}/restart" -o /dev/null -w "HTTP %{http_code}\n"
  else
    echo "==> No instance yet (status=${status}); waiting for allocation..."
  fi
else
  echo "==> Unknown status=${status}; trying start"
  curl -sf -X POST "${auth[@]}" "${cg_url}/start" -o /dev/null -w "HTTP %{http_code}\n" || true
fi

echo "==> Waiting for running instance with SSH..."
ssh_ip="" ssh_port=""
for i in $(seq 1 60); do
  inst_json="$(curl -sf "${auth[@]}" "${instances_url}")"
  ssh_ip="$(echo "${inst_json}" | jq -r '.instances[0].ssh_ip // empty')"
  ssh_port="$(echo "${inst_json}" | jq -r '.instances[0].ssh_port // empty')"
  state="$(echo "${inst_json}" | jq -r '.instances[0].state // empty')"
  ready="$(echo "${inst_json}" | jq -r '.instances[0].ready // false')"
  echo "  poll $i: state=${state} ready=${ready} ssh=${ssh_ip}:${ssh_port}"
  if [[ "${state}" == "running" && "${ready}" == "true" && -n "${ssh_ip}" && "${ssh_ip}" != "null" ]]; then
    break
  fi
  sleep 10
done

if [[ -z "${ssh_ip}" || "${ssh_ip}" == "null" ]]; then
  echo "ERROR: SSH not available" >&2
  exit 1
fi

echo "==> SSH root@${ssh_ip}:${ssh_port}"
sleep 20
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "root@${ssh_ip}" -p "${ssh_port}" \
  "cat /opt/propminer/VERSION 2>/dev/null; ps aux | grep propminer | grep -v grep | head -1" || true

echo "==> Tailing /tmp/propminer.log for ${WATCH_SECS}s..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "root@${ssh_ip}" -p "${ssh_port}" \
  "timeout ${WATCH_SECS} tail -F /tmp/propminer.log 2>/dev/null || sleep ${WATCH_SECS}" \
  | grep -E --line-buffered 'build=|stratum|submit|accepted|rejected|verify|gpu-hit|pool|shares:' || true

echo "==> Done."
