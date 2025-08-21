import socket
import time
import json, re

HOST, PORT = "127.0.0.1", 6000
TEAM_NAME = "TritonBot"
VERSION = 15
RAW_JSONL = "fullstate_raw.jsonl"

def frames_from_buffer(buf):
    frames, depth, start = [], 0, None
    for i,ch in enumerate(buf):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0 and start is not None:
                frames.append(buf[start:i+1])
                start = None
    remainder = "" if depth==0 else buf[start if start is not None else len(buf):]
    return frames, remainder

FULLSTATE_HDR = re.compile(r'^\(fullstate\s+(\d+)\s', re.S)
PMODE = re.compile(r'\(pmode\s+([^)]+)\)')
BALL_ALT1 = re.compile(r'\(\(b\)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)')  # ((b) x y vx vy)
BALL_ALT2 = re.compile(r'\(ball\s+\(pos\s+([-\d.]+)\s+([-\d.]+)\)\s+\(vel\s+([-\d.]+)\s+([-\d.]+)\)\)')

def extract_ball(s):
    m = BALL_ALT1.search(s) or BALL_ALT2.search(s)
    if m:
        x,y,vx,vy = map(float, m.groups())
        return {"ball_x":x, "ball_y":y, "ball_vx":vx, "ball_vy":vy}
    return {}

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    init_msg = f"(init {TEAM_NAME} (version {VERSION}))\n"
    s.sendall(init_msg.encode("utf-8"))

    buf = ""
    with open(RAW_JSONL, "w") as out:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="ignore")
            frames, buf = frames_from_buffer(buf)
            for f in frames:
                if f.startswith("(fullstate"):
                    cycle = None
                    m = FULLSTATE_HDR.match(f)
                    if m:
                        cycle = int(m.group(1))
                    pm = PMODE.search(f)
                    pmode = pm.group(1) if pm else None
                    ball = extract_ball(f)

                    rec = {
                        "ts": time.time(),
                        "cycle": cycle,
                        "pmode": pmode,
                        "raw": f, 
                        **ball
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()

if __name__ == "__main__":
    main()
