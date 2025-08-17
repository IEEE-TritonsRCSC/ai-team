import re
import time
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 6000
SERVER_ADDR = (UDP_IP, UDP_PORT)
INIT_PATTERN = r"\(init ([lr]) (1[0-1]|[1-9]) before_kick_off\)"

# UDP client to send a message to the simulation server
class TritonClient:
    def __init__(self, teamname: str, id: int, pose: tuple):
        self.id = id
        self.pose = pose
        self.count = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect(teamname.encode())

    def main(self):
        print("I am online")
        time.sleep(15)
        if self.count < 135:
            self.count += 1
            if self.count % 2 == 0:
                self.sock.sendto(b"(dash 100)\0", self.addr)
            else:
                self.sock.sendto(b"(kick 100 0)\0", self.addr)
            time.sleep(0.1)
        print("Going offline")
        self.disconnect()
        print("done")

    def get_action(self, ball_pos: tuple, poses: list[tuple]):
        self.pose = poses[self.id]
        if self.count < 135:
            action = "dash 100" if self.count % 2 == 0 else "kick 100 0"
        elif self.count == 135:
            action = "bye"
        else:
            action = None
        self.count += 1
        return action

    def execute_action(self, action: str):
        if action:
            self.sock.sendto(f"({action})\0".encode(), self.addr)

    def connect(self, teamname: bytes):
        self.sock.sendto(b"(init %b (version 19))\0" % teamname, SERVER_ADDR)
        (data, address) = self.sock.recvfrom(64)
        if m := re.search(INIT_PATTERN, data.decode()):
            self.side = m.group(1)
            self.id = int(m.group(2))
            self.addr = address    # Save the address for later use
        else:
            raise Exception(f"Unexpected response: {data} from {address}")
        
        time.sleep(0.1)
        tmp_args = f"{self.pose[0]} {self.pose[1]}".encode()
        self.sock.sendto(b"(move %b)\0" % tmp_args, self.addr)
        time.sleep(0.1)
        self.sock.sendto(b"(turn %b)\0" % str(self.pose[2]).encode(), self.addr)

    def disconnect(self):
        self.sock.sendto(b"(bye)\0", self.addr)


if __name__ == "__main__":
    client = TritonClient("Triton", 1, (-9.5, 0.0, 0.0))
    client.main()
