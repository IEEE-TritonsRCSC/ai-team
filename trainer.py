import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 6001    # trainer port
SERVER_ADDR = (UDP_IP, UDP_PORT)

class Trainer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, 0))

        self.sock.sendto(b"(init (version 19))\0", SERVER_ADDR)
        (data, address) = self.sock.recvfrom(1024)
        if data != b"(init ok)\0":
            raise Exception(f"Unexpected response: {data} from {address}")
        else:
            self.addr = address    # Save the address for later use

    def main(self):
        self.sock.sendto(b"(eye on)\0", self.addr)
        while True:
            (data, address) = self.sock.recvfrom(1024)
            if address == self.addr:
                print(f"Received message: {data} from {address}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()
