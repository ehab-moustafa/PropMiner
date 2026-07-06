#!/usr/bin/env bash
# Download PropMiner binaries from GitHub Releases (SRBMiner-style fast deploy).
#
# Env:
#   PROPMINER_USE_RELEASE=1          — enable release download (bootstrap image default)
#   PROPMINER_RELEASE_REPO         — default ehab-moustafa/PropMiner
#   PROPMINER_RELEASE_TAG          — default continuous (rolling latest)
#   PROPMINER_RELEASE_ASSET          — default propminer-rtx5090-linux-amd64.tar.gz
#   PROPMINER_RELEASE_URL            — override full download URL
#   PROPMINER_GITHUB_TOKEN           — optional, for private repos
#   PROPMINER_RELEASE_CACHE          — cache dir (default ~/.cache/propminer/releases)
#   PROPMINER_AUTO_UPDATE=1          — re-check release before each mine restart
set -euo pipefail

: "${ROOT:?ROOT must be set before sourcing download_release.sh}"

RELEASE_REPO="${PROPMINER_RELEASE_REPO:-ehab-moustafa/PropMiner}"
RELEASE_TAG="${PROPMINER_RELEASE_TAG:-continuous}"
RELEASE_ASSET="${PROPMINER_RELEASE_ASSET:-propminer-rtx5090-linux-amd64.tar.gz}"
RELEASE_CACHE="${PROPMINER_RELEASE_CACHE:-${HOME}/.cache/propminer/releases}"

release_log() {
    echo "[release] $*" | propminer_log
}

_release_curl() {
    local url="$1"
    local out="$2"
    local auth=()
    if [[ -n "${PROPMINER_GITHUB_TOKEN:-}" ]]; then
        auth=(-H "Authorization: Bearer ${PROPMINER_GITHUB_TOKEN}")
    fi
    curl -fsSL "${auth[@]}" -o "${out}" "${url}"
}

_resolve_download_url() {
  local tag="$1"
  if [[ -n "${PROPMINER_RELEASE_URL:-}" ]]; then
    printf '%s' "${PROPMINER_RELEASE_URL}"
    return 0
  fi
  python3 - "${RELEASE_REPO}" "${tag}" "${RELEASE_ASSET}" <<'PY'
import json
import sys
import urllib.request

repo, tag, asset_name = sys.argv[1:4]
api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.load(resp)
for asset in data.get("assets", []):
    if asset.get("name") == asset_name:
        print(asset["browser_download_url"])
        break
else:
    raise SystemExit(f"asset not found: {asset_name} in release {tag}")
PY
}

_remote_version() {
    local url="$1"
    local tmp
    tmp="$(mktemp)"
    _release_curl "${url}" "${tmp}"
    tar -xOzf "${tmp}" VERSION 2>/dev/null || echo "unknown"
    rm -f "${tmp}"
}

_install_release_tar() {
    local archive="$1"
    local build_dir="$2"
    local root_dir="$3"
    mkdir -p "${build_dir}"
    tar -xzf "${archive}" -C "${build_dir}"
    for f in propminer libpearl_gemm_capi.so libpearl_mining_capi.so; do
        if [[ ! -f "${build_dir}/${f}" ]]; then
            release_log "ERROR: missing ${f} in release archive"
            return 1
        fi
    done
    chmod +x "${build_dir}/propminer"
    cp -f "${build_dir}/propminer" "${build_dir}/libpearl_gemm_capi.so" \
        "${build_dir}/libpearl_mining_capi.so" "${root_dir}/"
    if [[ -f "${build_dir}/VERSION" ]]; then
        cp -f "${build_dir}/VERSION" "${root_dir}/VERSION"
        cp -f "${build_dir}/VERSION" "${RELEASE_CACHE}/installed.version"
    fi
    release_log "installed version=$(cat "${build_dir}/VERSION" 2>/dev/null || echo unknown)"
    return 0
}

download_release_binaries() {
    local build_dir="${1:-${ROOT}/build_remote_test}"
    local root_dir="${2:-${ROOT}}"
    mkdir -p "${RELEASE_CACHE}" "${build_dir}"

    local url
    url="$(_resolve_download_url "${RELEASE_TAG}")"
    release_log "repo=${RELEASE_REPO} tag=${RELEASE_TAG}"
    release_log "url=${url}"

    local cached_archive="${RELEASE_CACHE}/${RELEASE_TAG}-${RELEASE_ASSET}"
    local need_download=1
    if [[ -f "${cached_archive}" && -f "${RELEASE_CACHE}/installed.version" \
        && -x "${build_dir}/propminer" ]]; then
        local remote_ver
        remote_ver="$(_remote_version "${url}" 2>/dev/null || true)"
        if [[ -n "${remote_ver}" && "$(cat "${RELEASE_CACHE}/installed.version")" == "${remote_ver}" ]]; then
            need_download=0
            release_log "cache hit version=${remote_ver}"
        fi
    fi

    if [[ "${need_download}" -eq 1 ]]; then
        release_log "downloading ${RELEASE_ASSET}..."
        _release_curl "${url}" "${cached_archive}"
    fi
    _install_release_tar "${cached_archive}" "${build_dir}" "${root_dir}"

    if [[ -x "${build_dir}/propminer" ]]; then
        release_log "propminer ready in ${build_dir}"
        return 0
    fi
    release_log "ERROR: propminer missing after install"
    return 1
}

maybe_auto_update_release() {
    [[ "${PROPMINER_AUTO_UPDATE:-0}" == "1" ]] || return 0
    [[ "${PROPMINER_USE_RELEASE:-0}" == "1" ]] || return 0
    local build_dir="${ROOT}/build_remote_test"
    local url
    url="$(_resolve_download_url "${RELEASE_TAG}")" || return 0
    local remote_ver local_ver
    remote_ver="$(_remote_version "${url}" 2>/dev/null || true)"
    local_ver="$(cat "${ROOT}/VERSION" 2>/dev/null || true)"
    if [[ -n "${remote_ver}" && "${remote_ver}" != "${local_ver}" ]]; then
        release_log "auto-update ${local_ver:-none} -> ${remote_ver}"
        download_release_binaries "${build_dir}" "${ROOT}" || true
    fi
}

ensure_binaries() {
    local build_dir="${1:-${ROOT}/build_remote_test}"
    if [[ "${PROPMINER_USE_RELEASE:-0}" == "1" ]]; then
        download_release_binaries "${build_dir}" "${ROOT}"
    else
        prepare_prebuilt_binaries "${build_dir}"
    fi
}
