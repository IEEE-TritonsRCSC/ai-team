import re
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 6001    # trainer port
SERVER_ADDR = (UDP_IP, UDP_PORT)
NUM_PLAYERS = 2    # to be configured

class Trainer:
    def __init__(self, teamname: str, n_players: int):
        self.teamname = teamname
        self.n_players = n_players
        self.ball_pos = (0.0, 0.0)    # Default ball position
        # because unum is 1-indexed, first element is dummy
        self.poses = {self.teamname: [None] * (n_players + 1)}
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
                self.extract_player_pose(message)
                # Print the extracted information
                print(f"TIME: {server_time} | Ball position: {self.ball_pos}" +
                      f" | Player positions: {self.poses[self.teamname][1:]}")


    def extract_ball_pos(self, msg: str):
        """Extract ball position from the message."""
        m = re.search(r" \(\(b\) ", msg)
        if m:
            ball_xy = msg[m.end():].split()[0:2]
            self.ball_pos = tuple(float(coordinate) for coordinate in ball_xy)


    def extract_player_pose(self, msg: str):
        """Extract player pose from the message."""
        m = re.search(r"\(\(p \"(\w*)\" (1[0-1]|[1-9])\) ", msg)
        while m:
            team, unum = m.group(1), int(m.group(2))
            if team == self.teamname and unum <= self.n_players:    # check
                description = msg[m.end():].split()
                # pose (x, y, theta) information is in indices 0, 1, 4
                pose = tuple(float(description[i]) for i in [0, 1, 4])
                self.poses[team][unum] = pose
            
            # Move past the current match
            msg = msg[m.end():]
            m = re.search(r"\(\(p \"(\w*)\" (1[0-1]|[1-9])\) ", msg)


if __name__ == "__main__":
    trainer = Trainer("TritonBot", NUM_PLAYERS)
    trainer.main()
