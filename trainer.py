import re
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 6001    # trainer port
SERVER_ADDR = (UDP_IP, UDP_PORT)

class Trainer:
    def __init__(self):
        self.ball_pos = None    # placeholder for ball position
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect()    # activate connection with server

    def connect(self):
        # first contact the server to establish a connection
        self.sock.bind((UDP_IP, 0))
        self.sock.sendto(b"(init (version 19))\0", SERVER_ADDR)
        (data, address) = self.sock.recvfrom(16)
        if data != b"(init ok)\0":
            raise Exception(f"Unexpected response: {data} from {address}")
        else:
            self.addr = address    # Save address for subsequent communication
            self.sock.sendto(b"(eye on)\0", self.addr)

        # skip initialization messages
        while True:
            (data, address) = self.sock.recvfrom(16)
            if address == self.addr and data == b"(ok eye on)\0":
                print("Trainer is now active.")
                break

    def main(self):
        while True:
            (data, address) = self.sock.recvfrom(256)
            if address == self.addr:
                message = data.decode()
                # Extract information
                server_time = int(re.search(r"\d+", message).group())
                self.extract_ball_pos(message)
                # Print the extracted information
                print(f"TIME: {server_time} | Ball position: {self.ball_pos}")

    def extract_ball_pos(self, msg: str):
        """Extract ball position from the message."""
        m = re.search(r" \(\(b\) ", msg)
        if m:
            ball_xy = msg[m.end():].split()[0:2]
            self.ball_pos = tuple(float(coordinate) for coordinate in ball_xy)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()
