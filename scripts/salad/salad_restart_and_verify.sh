#!/usr/bin/env bash
# Restart prop-miner-official on SaladCloud and tail logs for share acceptance.
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

instances_url="${API_BASE}/organizations/${ORG}/projects/${PROJECT}/containers/${CONTAINER}/instances"
echo "==> Listing instances: ${instances_url}"
instances_json="$(curl -sf "${auth[@]}" "${instances_url}")"
instance_id="$(echo "${instances_json}" | jq -r '.instances[0].id // empty')"
if [[ -z "${instance_id}" ]]; then
  echo "ERROR: no running instance found" >&2
  echo "${instances_json}" | jq .
  exit 1
fi
echo "==> Instance: ${instance_id}"

restart_url="${instances_url}/${instance_id}/restart"
echo "==> POST restart: ${restart_url}"
curl -sf -X POST "${auth[@]}" "${restart_url}" -o /dev/null -w "HTTP %{http_code}\n"

echo "==> Waiting ${WATCH_SECS}s for miner logs (grep share/stratum/accepted)..."
sleep 30

if [[ -n "${SALAD_SSH:-}" ]]; then
  # SALAD_SSH="root@host -p port" — set after listing instance ssh_ip/ssh_port from API
  ssh_ip="$(echo "${instances_json}" | jq -r '.instances[0].ssh_ip // empty')"
  ssh_port="$(echo "${instances_json}" | jq -r '.instances[0].ssh_port // 22')"
  if [[ -n "${ssh_ip}" && "${ssh_ip}" != "null" ]]; then
    SALAD_SSH="root@${ssh_ip} -p ${ssh_port}"
  fi
fi

if [[ -n "${SALAD_SSH:-}" ]]; then
  echo "==> SSH log tail via ${SALAD_SSH}"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 ${SALAD_SSH} \
    "timeout ${WATCH_SECS} tail -F /opt/propminer/logs/propminer.log 2>/dev/null || \
     timeout ${WATCH_SECS} journalctl -u propminer -f 2>/dev/null || \
     timeout ${WATCH_SECS} tail -F /var/log/propminer.log" \
    | grep -E --line-buffered 'build=|stratum|submit|accepted|rejected|verify|gpu-hit|pool' || true
else
  echo "NOTE: Set SALAD_SSH or ensure instance exposes ssh_ip for live log tail."
  echo "Poll instance state:"
  for i in $(seq 1 12); do
    curl -sf "${auth[@]}" "${instances_url}" | jq '.instances[0] | {id, state, ready, version}'
    sleep 15
  done
fi

echo "==> Done. Grep locally saved logs for 'share accepted' or 'pool-no-response'."
