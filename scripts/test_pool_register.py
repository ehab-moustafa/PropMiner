#!/usr/bin/env python3
"""Local pool connectivity test — no GPU, no CUDA, no Salad required.

Exercises the same path as production mine mode up to Register:
  TCP -> TLS (+ optional ALPN h2) -> HTTP/2 -> gRPC Register

Safe to run from your laptop before pushing images. Uses your real wallet
worker name but does not start mining or submit shares.

Usage:
  PROPMINER_WALLET=krxX2P3Z84 PROPMINER_WORKER=prop-miner \\
    python3 scripts/test_pool_register.py

  # Compare old client behavior (no ALPN, application/grpc):
  python3 scripts/test_pool_register.py --legacy

  # Other pool/host:
  PROPMINER_POOL=prl-eu.kryptex.network:443 python3 scripts/test_pool_register.py
"""
from __future__ import annotations

import argparse
import socket
import ssl
import struct
import sys
import os
import time

FRAME_HEADER_SIZE = 9
FRAME_DATA = 0x0
FRAME_HEADERS = 0x1
FRAME_SETTINGS = 0x4
FRAME_GOAWAY = 0x7
FRAME_WINDOW_UPDATE = 0x8
FLAG_END_STREAM = 0x1
FLAG_ACK = 0x1
FLAG_END_HEADERS = 0x4


def parse_host_port(pool: str) -> tuple[str, int]:
    if ":" in pool:
        host, port_s = pool.rsplit(":", 1)
        return host, int(port_s)
    return pool, 443


def split_wallet_worker(wallet: str, worker: str) -> tuple[str, str]:
    if worker or "." not in wallet:
        return wallet, worker or "propminer"
    base, wrk = wallet.split(".", 1)
    return base, wrk


def pb_varint(v: int) -> bytes:
    out = bytearray()
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def pb_string(field: int, s: str) -> bytes:
    b = s.encode("utf-8")
    tag = pb_varint((field << 3) | 2)
    return tag + pb_varint(len(b)) + b


def pb_uint32(field: int, v: int) -> bytes:
    tag = pb_varint((field << 3) | 0)
    return tag + pb_varint(v)


def pb_embedded(field: int, msg: bytes) -> bytes:
    tag = pb_varint((field << 3) | 2)
    return tag + pb_varint(len(msg)) + msg


def encode_gpu_card(uuid: str, model: str, index: int) -> bytes:
    parts = [
        pb_string(1, uuid),
        pb_string(2, model),
        pb_uint32(3, index),
    ]
    return b"".join(parts)


def encode_register_request(wallet: str, worker: str, k: int = 128) -> bytes:
    gpu = encode_gpu_card("local-test-gpu-00000000", "RTX 5090 (pool-test)", 0)
    parts = [
        pb_string(3, wallet),
        pb_string(4, worker),
        pb_embedded(5, gpu),
        pb_string(7, "propminer/pool-test"),
        pb_uint32(9, 2),
        pb_uint32(10, k),
    ]
    return b"".join(parts)


def hpack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) < 128:
        return bytes([len(b)]) + b
    raise ValueError("hpack string too long for this test harness")


def hpack_literal(name: str, value: str) -> bytes:
    return bytes([0x40]) + hpack_string(name) + hpack_string(value)


def encode_request_headers(
    authority: str,
    path: str,
    user_agent: str,
    content_type: str,
) -> bytes:
    parts = [
        hpack_literal(":method", "POST"),
        hpack_literal(":scheme", "https"),
        hpack_literal(":authority", authority),
        hpack_literal(":path", path),
        hpack_literal("content-type", content_type),
        hpack_literal("user-agent", user_agent),
        hpack_literal("te", "trailers"),
    ]
    return b"".join(parts)


def write_u24_be(length: int) -> bytes:
    return struct.pack(">I", length)[1:4]


def h2_frame(ftype: int, flags: int, stream_id: int, payload: bytes = b"") -> bytes:
    hdr = write_u24_be(len(payload))
    hdr += bytes([ftype, flags])
    hdr += struct.pack(">I", stream_id & 0x7FFFFFFF)
    return hdr + payload


def read_exact(sock: ssl.SSLSocket, n: int, timeout: float) -> bytes:
    sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed while reading")
        buf.extend(chunk)
    return bytes(buf)


def pb_decode_register_response(data: bytes) -> dict:
    out: dict = {}
    i = 0
    while i < len(data):
        tag, i = _read_varint(data, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _read_varint(data, i)
            if field == 1:
                out["success"] = bool(val)
            elif field == 5:
                out["initial_difficulty_nbits"] = val
        elif wire == 2:
            ln, i = _read_varint(data, i)
            val = data[i : i + ln]
            i += ln
            if field == 2 and ln == 16:
                out["miner_id"] = val.hex()
            elif field == 3:
                out["session_token"] = val.decode("utf-8", errors="replace")
            elif field == 6 and ln == 32:
                out["pool_id"] = val.hex()
            elif field == 7:
                out["error_message"] = val.decode("utf-8", errors="replace")
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            # skip unknown / deprecated wire types
            if wire == 3 or wire == 4:
                raise ValueError("group wire types not supported in proto3")
            i = _skip_unknown(data, i, wire)
    return out


def _skip_unknown(data: bytes, i: int, wire: int) -> int:
    if wire == 0:
        while i < len(data) and (data[i] & 0x80):
            i += 1
        return i + 1
    if wire == 1:
        return i + 8
    if wire == 2:
        ln, i = _read_varint(data, i)
        return i + ln
    if wire == 5:
        return i + 4
    raise ValueError(f"unsupported wire type {wire}")


def _read_varint(data: bytes, i: int) -> tuple[int, int]:
    v = 0
    shift = 0
    while i < len(data):
        b = data[i]
        i += 1
        v |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return v, i
        shift += 7
    raise ValueError("truncated varint")


def http2_handshake(tls: ssl.SSLSocket, timeout: float, *, legacy_window_update: bool = False) -> None:
    tls.sendall(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
    tls.sendall(h2_frame(FRAME_SETTINGS, 0, 0))
    if legacy_window_update:
        # v1.43 bug: overflows connection flow-control window -> GOAWAY(0x3).
        tls.sendall(h2_frame(FRAME_WINDOW_UPDATE, 0, 0, struct.pack(">I", 0x7FFFFFFF)))

    deadline = time.time() + timeout
    got_settings = got_ack = False
    while not (got_settings and got_ack):
        if time.time() > deadline:
            raise TimeoutError("HTTP/2 handshake timeout")
        hdr = read_exact(tls, FRAME_HEADER_SIZE, max(0.1, deadline - time.time()))
        length = struct.unpack(">I", b"\x00" + hdr[0:3])[0]
        ftype, flags = hdr[3], hdr[4]
        sid = struct.unpack(">I", hdr[5:9])[0] & 0x7FFFFFFF
        payload = read_exact(tls, length, max(0.1, deadline - time.time())) if length else b""
        if ftype == FRAME_SETTINGS:
            if flags & FLAG_ACK:
                got_ack = True
            else:
                got_settings = True
                tls.sendall(h2_frame(FRAME_SETTINGS, FLAG_ACK, 0))
        elif ftype == FRAME_GOAWAY:
            code = struct.unpack(">I", payload[4:8])[0] if len(payload) >= 8 else -1
            raise ConnectionError(f"GOAWAY during HTTP/2 handshake (error_code={code})")
        elif ftype == FRAME_WINDOW_UPDATE and sid == 0:
            pass


def grpc_register(
    host: str,
    port: int,
    wallet: str,
    worker: str,
    *,
    use_alpn: bool,
    content_type: str,
    timeout: float,
    legacy_window_update: bool = False,
) -> dict:
    ctx = ssl.create_default_context()
    if use_alpn:
        ctx.set_alpn_protocols(["h2"])

    raw = socket.create_connection((host, port), timeout=timeout)
    tls = ctx.wrap_socket(raw, server_hostname=host)
    alpn = tls.selected_alpn_protocol()

    http2_handshake(tls, timeout, legacy_window_update=legacy_window_update)

    stream_id = 1
    path = "/pearlpool.mining.v2.MinerService/Register"
    authority = f"{host}:{port}"
    headers = encode_request_headers(
        authority, path, "propminer/pool-test", content_type
    )
    tls.sendall(
        h2_frame(
            FRAME_HEADERS,
            FLAG_END_HEADERS,
            stream_id,
            headers,
        )
    )

    body = encode_register_request(wallet, worker)
    grpc_msg = b"\x00" + struct.pack(">I", len(body)) + body
    tls.sendall(h2_frame(FRAME_DATA, FLAG_END_STREAM, stream_id, grpc_msg))

    deadline = time.time() + timeout
    response_body = bytearray()
    while time.time() < deadline:
        hdr = read_exact(tls, FRAME_HEADER_SIZE, max(0.1, deadline - time.time()))
        length = struct.unpack(">I", b"\x00" + hdr[0:3])[0]
        ftype, flags = hdr[3], hdr[4]
        rsid = struct.unpack(">I", hdr[5:9])[0] & 0x7FFFFFFF
        payload = (
            read_exact(tls, length, max(0.1, deadline - time.time())) if length else b""
        )

        if ftype == FRAME_GOAWAY:
            code = struct.unpack(">I", payload[4:8])[0] if len(payload) >= 8 else -1
            raise ConnectionError(f"GOAWAY during register (error_code={code})")

        if rsid != stream_id:
            continue

        if ftype == FRAME_DATA:
            response_body.extend(payload)
            if flags & FLAG_END_STREAM:
                break

    tls.close()

    if len(response_body) < 5:
        raise RuntimeError(f"empty register response ({len(response_body)} bytes)")

    # gRPC framing: 1 byte compression + 4 byte length + protobuf
    if response_body[0:1] == b"<" or response_body[:5] == b"\x00" and len(response_body) > 5:
        preview = bytes(response_body[:120])
        if preview.lstrip().startswith(b"<"):
            raise RuntimeError(
                "non-gRPC HTML response (likely 503 from load balancer): "
                + preview.decode("utf-8", errors="replace")[:200]
            )
    msg_len = struct.unpack(">I", response_body[1:5])[0]
    proto = bytes(response_body[5 : 5 + msg_len])
    decoded = pb_decode_register_response(proto)
    decoded["alpn"] = alpn
    decoded["content_type"] = content_type
    return decoded


def main() -> int:
    ap = argparse.ArgumentParser(description="Test Kryptex gRPC Register locally")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="Old client: no ALPN, content-type application/grpc",
    )
    ap.add_argument(
        "--pool",
        default=os.environ.get("PROPMINER_POOL", "prl.kryptex.network:443"),
    )
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    wallet = os.environ.get("PROPMINER_WALLET", "")
    worker = os.environ.get("PROPMINER_WORKER", "")
    if not wallet:
        print("ERROR: set PROPMINER_WALLET (and optionally PROPMINER_WORKER)", file=sys.stderr)
        return 1

    wallet, worker = split_wallet_worker(wallet, worker)
    host, port = parse_host_port(args.pool)

    mode = "legacy (no ALPN, application/grpc)" if args.legacy else "fixed (ALPN h2, application/grpc+proto)"
    print(f"Pool test mode: {mode}")
    print(f"Target: {host}:{port}")
    print(f"Wallet: {wallet}  Worker: {worker}")
    print()

    try:
        result = grpc_register(
            host,
            port,
            wallet,
            worker,
            use_alpn=not args.legacy,
            content_type="application/grpc" if args.legacy else "application/grpc+proto",
            timeout=args.timeout,
            legacy_window_update=args.legacy,
        )
    except Exception as e:
        print(f"FAIL: {e}")
        return 2

    print(f"ALPN negotiated: {result.get('alpn')!r}")
    print(f"Content-Type sent: {result.get('content_type')}")
    if result.get("success"):
        print("PASS: Register accepted by pool")
        if result.get("miner_id"):
            print(f"  miner_id: {result['miner_id']}")
        if result.get("session_token"):
            tok = result["session_token"]
            print(f"  session_token: {tok[:12]}... ({len(tok)} chars)")
        if result.get("initial_difficulty_nbits") is not None:
            print(f"  initial_difficulty_nbits: {result['initial_difficulty_nbits']}")
        return 0

    print("FAIL: Register rejected by pool")
    print(f"  error_message: {result.get('error_message', '<none>')}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
