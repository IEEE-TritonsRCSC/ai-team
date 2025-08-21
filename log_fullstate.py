import socket, time, json, re, sys

HOST, PORT = "127.0.0.1", 6000        # UDP player port
TEAM_NAME = "TritonBot"
VERSION = 19
RAW_JSONL = "fullstate_raw.jsonl"

FULLSTATE_HDR = re.compile(r'^\(fullstate\s+(\d+)\s', re.S)
PMODE = re.compile(r'\(pmode\s+([^)]+)\)')
BALL_ALT1 = re.compile(r'\(\(b\)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)')
BALL_ALT2 = re.compile(r'\(ball\s+\(pos\s+([-\d.]+)\s+([-\d.]+)\)\s+\(vel\s+([-\d.]+)\s+([-\d.]+)\)\)')

def extract_ball(s):
    m = BALL_ALT1.search(s) or BALL_ALT2.search(s)
    if m:
        x,y,vx,vy = map(float, m.groups())
        return {"ball_x":x, "ball_y":y, "ball_vx":vx, "ball_vy":vy}
    return {}

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    server_addr = (HOST, PORT)

    # 1) INIT
    init_msg = f"(init {TEAM_NAME} (version {VERSION}))\0".encode()
    s.sendto(init_msg, server_addr)

    init_ok = False
    addr = server_addr
    t0 = time.time()
    while time.time() - t0 < 5.0:   # try for up to 5s
        try:
            data, addr = s.recvfrom(4096)
            text = data.decode("utf-8", errors="ignore")
            # Expect: (init l|r <unum> before_kick_off)
            if text.startswith("(init "):
                print("[init] <-", text.strip())
                init_ok = True
                break
            else:
                print("[init] got non-init:", text[:120].replace("\n"," "))
        except socket.timeout:
            # re-send init periodically in case server didn’t catch the first one
            s.sendto(init_msg, server_addr)

    if not init_ok:
        print("ERROR: No init reply. Is rcssserver running on UDP:6000?")
        sys.exit(1)

    # 2) Pre-kickoff pose (optional)
    for cmd in [b"(move -9.5 0)\0", b"(turn 0)\0"]:
        s.sendto(cmd, addr)
        time.sleep(0.05)

    # 3) Receive loop: log any packets; write out fullstate frames if present
    print("[recv] listening for packets (press Ctrl+C to stop)…")
    rec_count = 0
    with open(RAW_JSONL, "w") as out:
        try:
            while True:
                try:
                    data, _ = s.recvfrom(65536)
                except socket.timeout:
                    # Keep the socket alive by sending a harmless noop action occasionally
                    s.sendto(b"(turn 0)\0", addr)
                    continue

                msg = data.decode("utf-8", errors="ignore")
                rec_count += 1

                # Print minimal live feedback (first 1–2 lines)
                if rec_count <= 5 or rec_count % 50 == 0:
                    print(f"[recv {rec_count}] {msg.splitlines()[0][:120]}")

                if msg.startswith("(fullstate"):
                    m = FULLSTATE_HDR.match(msg)
                    cycle = int(m.group(1)) if m else None
                    pm = PMODE.search(msg)
                    pmode = pm.group(1) if pm else None
                    ball = extract_ball(msg)
                    rec = {
                        "ts": time.time(),
                        "cycle": cycle,
                        "pmode": pmode,
                        "raw": msg,
                        **ball
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
        except KeyboardInterrupt:
            pass
        finally:
            s.sendto(b"(bye)\0", addr)

if __name__ == "__main__":
    main()

