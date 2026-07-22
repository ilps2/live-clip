#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""danmaku.py — B 站直播弹幕抓取（websocket 公开协议）。

用法:
    python danmaku.py --room 31654967 --duration 600 --out results/danmaku.jsonl

输出 JSONL 每行:
    {"ts": 1700000000.123, "rel_t": 12.3, "type": "DANMU_MSG",
     "user": "xxx", "text": "弹幕内容"}
rel_t 为相对抓取开始的秒数（与录制/分数时间轴对齐用）。

协议参考（公开文档化）:
    - 连接: wss://broadcastlv.chat.bilibili.com:2245/sub
    - 包头 16B: packetLen(4) headerLen(2) ver(2) op(4) seq(4)
    - ver=1 纯文本, ver=2 zlib deflate, ver=3 brotli
    - op: 7=进房, 8=进房回复, 2=心跳, 3=心跳回复(人气值), 5=命令(弹幕等)
"""
import argparse
import hashlib
import json
import struct
import time
import urllib.parse
import urllib.request
import zlib

import random

import websocket  # websocket-client

try:
    import brotli
except ImportError:
    brotli = None

WS_URL = "wss://broadcastlv.chat.bilibili.com:2245/sub"  # 无 token 时的兜底
HEADER_LEN = 16
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/120 Safari/537.36"}
_WBI_MIXIN = [46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,
              33,9,42,19,29,28,14,39,12,38,41,13,37,48,7,16,24,55,40,
              61,26,17,0,1,60,51,30,4,22,25,54,21,56,59,6,63,57,62,11,
              36,20,34,44,52]


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=10))


def _wbi_sign(params: dict) -> dict:
    """B 站 WBI 签名（匿名即可），getDanmuInfo 等接口的风控要求。"""
    nav = _http_json("https://api.bilibili.com/x/web-interface/nav")
    wbi = nav["data"]["wbi_img"]
    key = (wbi["img_url"].rsplit("/", 1)[1].split(".")[0]
           + wbi["sub_url"].rsplit("/", 1)[1].split(".")[0])
    mixin = "".join(key[i] for i in _WBI_MIXIN)[:32]
    params = dict(params)
    params["wts"] = int(time.time())
    q = urllib.parse.urlencode(sorted(params.items()))
    params["w_rid"] = hashlib.md5((q + mixin).encode()).hexdigest()
    return params


def get_danmu_auth(room_id: int) -> tuple[str, str, str]:
    """返回 (ws_url, token, buvid3)。风控下裸连会被秒断：必须 WBI 签名
    拿 token，且 ws 握手 Cookie 里带 buvid3（spi 接口匿名可领）。"""
    spi = _http_json("https://api.bilibili.com/x/frontend/finger/spi")
    buvid3 = spi["data"]["b_3"]
    params = _wbi_sign({"id": room_id, "type": 0})
    req = urllib.request.Request(
        "https://api.live.bilibili.com/xlive/web-room/v1/index/"
        "getDanmuInfo?" + urllib.parse.urlencode(params),
        headers={**UA, "Cookie": f"buvid3={buvid3}",
                 "Referer": f"https://live.bilibili.com/{room_id}"})
    data = json.load(urllib.request.urlopen(req, timeout=10))["data"]
    host = data["host_list"][0]
    return f"wss://{host['host']}:{host['wss_port']}/sub", data["token"], buvid3


def pack_packet(payload: bytes, op: int, ver: int = 1) -> bytes:
    header = struct.pack(">IHHII", HEADER_LEN + len(payload), HEADER_LEN, ver, op, 1)
    return header + payload


def unpack_packets(buf: bytes):
    """解析一段二进制流，yield (op, ver, body)。"""
    off = 0
    while off + HEADER_LEN <= len(buf):
        plen, hlen, ver, op, _seq = struct.unpack(">IHHII", buf[off:off + HEADER_LEN])
        if plen <= 0 or off + plen > len(buf):
            break
        body = buf[off + hlen:off + plen]
        off += plen
        if ver == 2:
            body = zlib.decompress(body)
            yield from unpack_packets(body)
        elif ver == 3 and brotli is not None:
            body = brotli.decompress(body)
            yield from unpack_packets(body)
        else:
            yield op, ver, body


def iter_danmaku(room_id: int, stop_at: float):
    """连接房间并持续产出解析后的弹幕/事件 dict，直到 stop_at。"""
    try:
        ws_url, token, buvid3 = get_danmu_auth(room_id)
        uid = 0
    except Exception as e:
        print(f"[danmaku] getDanmuInfo 失败({e})，尝试裸连兜底", flush=True)
        ws_url, token, buvid3, uid = WS_URL, "", "", 0
    join_payload = json.dumps({
        "uid": uid, "roomid": int(room_id), "protover": 3,
        "platform": "web", "type": 2, "key": token,
    }).encode()

    headers = [f"User-Agent: {UA['User-Agent']}",
               "Origin: https://live.bilibili.com"]
    if buvid3:
        headers.append(f"Cookie: buvid3={buvid3}")
    ws = websocket.create_connection(ws_url, timeout=10, header=headers)
    ws.send_binary(pack_packet(join_payload, op=7))
    t0 = time.time()
    last_hb = t0
    print(f"[danmaku] joined room {room_id}", flush=True)

    while time.time() < stop_at:
        now = time.time()
        if now - last_hb >= 30:
            ws.send_binary(pack_packet(b"[object Object]", op=2))
            last_hb = now
        try:
            ws.settimeout(max(0.1, min(5, stop_at - now)))
            data = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not isinstance(data, bytes):
            continue
        for op, ver, body in unpack_packets(data):
            if op != 5:
                continue
            try:
                msg = json.loads(body.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            cmd = msg.get("cmd", "")
            if cmd == "DANMU_MSG":
                info = msg["info"]
                yield {
                    "ts": time.time(),
                    "type": "DANMU_MSG",
                    "user": info[2][1] if len(info) > 2 else "",
                    "text": info[1] if len(info) > 1 else "",
                }
            # 也可在此扩展 SUPER_CHAT_MESSAGE / SEND_GIFT 等信号


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=int, required=True, help="B站直播真实房间号")
    ap.add_argument("--duration", type=float, default=600, help="抓取秒数")
    ap.add_argument("--out", default="results/danmaku.jsonl")
    args = ap.parse_args()

    stop_at = time.time() + args.duration
    t0 = time.time()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        while time.time() < stop_at:
            try:
                for ev in iter_danmaku(args.room, stop_at):
                    ev["rel_t"] = round(time.time() - t0, 2)
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    if n % 100 == 0:
                        print(f"[danmaku] {n} msgs, {ev['rel_t']}s", flush=True)
            except Exception as e:  # 断线重连
                print(f"[danmaku] reconnect: {e}", flush=True)
                time.sleep(3)
    print(f"[danmaku] done: {n} messages in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
