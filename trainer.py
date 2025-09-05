import re
import time
import random
import socket
import argparse
from TritonClient import TritonClient

UDP_IP = "127.0.0.1"
UDP_PORT = 6001    # trainer port
SERVER_ADDR = (UDP_IP, UDP_PORT)

ESP_IP = "192.168.8.10"    # IP address of the ESP32 on LAN
ESP_PORT = 8888    # Arbitrary ESP port
ESP_ADDR = (ESP_IP, ESP_PORT)

NUM_PLAYERS = 2    # to be configured
MAX_RETRIES = 5    # maximum number of retries for player connections

parser = argparse.ArgumentParser()
parser.add_argument("-hardware", action="store_true", 
                    help="Additionally use hardware ESP32 for communication")

class Trainer:
    def __init__(self, teamname: str, n_players: int):
        # Initialize team name, player count, ball position, and player poses
        self.teamname = teamname
        self.n_players = n_players
        self.ball_pos = (0.0, 0.0)    # Default ball position
        # because unum is 1-indexed, first element is dummy
        self.poses = {self.teamname: [None] * (n_players + 1)}

        # Connect to the server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.connect()    # activate connection with server

        # Initialize retry count and connection status for players
        self.retry_count = 0
        self.all_players_connected = False
        self.players = self.init_players()

        if args.hardware:
            self.esp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    def init_players(self):
        players = [None] * (self.n_players + 1)    # dummy for 1-indexing
        for i in range(1, self.n_players + 1):
            if i == 1:    # Special first player
                players[i] = TritonClient(self.teamname, i, (-9.5, 0.0, 0.0))
            else:
                x, y = random.uniform(-30, -15), random.uniform(-10, 10)
                theta = random.uniform(-180, 180)
                players[i] = TritonClient(self.teamname, i, (x, y, theta))
        return players


    def main(self):
        # Ensure all players are connected before proceeding
        while not self.all_players_connected and self.retry_count < MAX_RETRIES:
            self.receive_information()
            self.retry_count += 1

        if not self.all_players_connected:
            raise Exception("Failed to connect all players.")
        else:
            print("All players connected.")
        
        # Start the game
        self.sock.sendto(b"(change_mode play_on)\0", self.addr)
        while True:
            self.receive_information()
            self.execute_actions(self.get_actions())

    def get_actions(self):
        actions = []
        for player in self.players[1::]:  # Skip dummy player
            action = player.get_action(self.ball_pos, 
                                       self.poses[self.teamname])
            actions.append(action)
        return actions
    
    def execute_actions(self, actions):
        if args.hardware:
            msg = ""
        for player, action in zip(self.players[1::], actions):
            player.execute_action(action)
            if args.hardware:
                if action is not None:
                    msg += "\n" + action
        if args.hardware and len(msg) > 0:
            msg += "\0"
            self.esp_sock.sendto(msg[1::].encode(), ESP_ADDR)

    def receive_information(self):
        """Receive information from the server."""
        (data, address) = self.sock.recvfrom(256)
        if address == self.addr:
            message = data.decode()
            # Extract information
            m = re.search(r"\d+", message)
            if m:
                server_time = int(m.group())
                self.extract_ball_pos(message)
                self.extract_player_pose(message)
                # Print the extracted information
                print(f"TIME: {server_time} | Ball Position: {self.ball_pos}" +
                        f" | Player Poses: {self.poses[self.teamname][1::]}")

    def extract_ball_pos(self, msg: str):
        """Extract ball position from the message."""
        m = re.search(r" \(\(b\) ", msg)
        if m:
            ball_xy = msg[m.end():].split()[0:2]
            self.ball_pos = tuple(float(coordinate) for coordinate in ball_xy)


    def extract_player_pose(self, msg: str):
        """Extract player pose from the message."""
        m = re.search(r"\(\(p \"(\w*)\" (1[0-1]|[1-9])\) ", msg)
        n_players_found = 0
        while m:
            team, unum = m.group(1), int(m.group(2))
            if team == self.teamname and unum <= self.n_players:    # check
                description = msg[m.end():].split()
                # pose (x, y, theta) information is in indices 0, 1, 4
                pose = tuple(float(description[i]) for i in [0, 1, 4])
                self.poses[team][unum] = pose
                n_players_found += 1

            # Move past the current match
            msg = msg[m.end():]
            m = re.search(r"\(\(p \"(\w*)\" (1[0-1]|[1-9])\) ", msg)

        if not self.all_players_connected and n_players_found == self.n_players:
            self.all_players_connected = True


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = Trainer("Triton", NUM_PLAYERS)
    trainer.main()
