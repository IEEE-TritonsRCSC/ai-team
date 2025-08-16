import re
import time
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 6000
SERVER_ADDR = (UDP_IP, UDP_PORT)
INIT_PATTERN = r"\(init ([lr]) ([1-9]|10|11) before_kick_off\)"

# UDP client to send a message to the simulation server
class TritonClient:
    def __init__(self, teamname: str, side: str, id: int, orientations: tuple):
        self.teamname = teamname
        self.side = side
        self.id = id
        self.orientations = orientations
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect()

    def main(self):
        print("I am online")
        self.sock.sendto(b"(move -9.5 0)\0", self.addr)
        time.sleep(15)
        for i in range(135):
            if i % 2 == 0:
                self.sock.sendto(b"(dash 100)\0", self.addr)
            else:
                self.sock.sendto(b"(kick 100 0)\0", self.addr)
            time.sleep(0.1)
        print("Going offline")
        self.disconnect()
        print("done")

    def connect(self):
        self.sock.sendto(b"(init TritonBot (version 19))\0", SERVER_ADDR)
        (data, address) = self.sock.recvfrom(64)
        if m := re.search(INIT_PATTERN, data.decode()):
            self.side = m.group(1)
            self.id = int(m.group(2))
            self.addr = address    # Save the address for later use
        else:
            raise Exception(f"Unexpected response: {data} from {address}")

    def disconnect(self):
        self.sock.sendto(b"(bye)\0", self.addr)

client = TritonClient("Triton", "left", 0, (0, 0, 0))

if __name__ == "__main__":
    client.main()
