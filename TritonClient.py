import socket
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 6000
UDP_CONFIG = (UDP_IP, UDP_PORT)

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
        self.sock.sendto(b"(move -9.5 0)", UDP_CONFIG)
        time.sleep(15)
        for i in range(100):
            if i % 2 == 0:
                self.sock.sendto(b"(dash 100)", UDP_CONFIG)
            else:
                self.sock.sendto(b"(kick 100 0)", UDP_CONFIG)
            time.sleep(0.1)
            #"""
        print("Going offline")
        self.disconnect()
        print("done")

    def connect(self):
        self.sock.sendto(b"(init TritonBot)", UDP_CONFIG)

    def disconnect(self):
        self.sock.sendto(b"(bye)", UDP_CONFIG)

client = TritonClient("Triton", "left", 0, (0, 0, 0))

if __name__ == "__main__":
    client.main()
