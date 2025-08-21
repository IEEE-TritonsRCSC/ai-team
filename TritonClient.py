### IMPORTS ###
import socket
import time
import math
import re
import multiprocessing
import trainer
from multiprocessing import Manager


### CONSTANTS ###
UDP_IP = "127.0.0.1"
UDP_PORT = 6000
UDP_CONFIG = (UDP_IP, UDP_PORT)
INIT_PATTERN = r"\(init ([lr]) ([1-9]|10|11) before_kick_off\)"


### GLOBAL TRAINER DATA ###
manager = Manager()
global_trainer_data = manager.dict()
global_trainer_data.update({
    'cycle': 0,
    'ball_x': 0.0,
    'ball_y': 0.0,
    'ball_speed_x': 0.0,
    'ball_speed_y': 0.0,
    'players': {},
    'goals': {'left': {'x': -52.5, 'y': 0}, 'right': {'x': 52.5, 'y': 0}}
})


### GETTERS ###
def get_cycle():
    return global_trainer_data['cycle']

def get_ball_position():
    return (global_trainer_data['ball_x'], global_trainer_data['ball_y'])

def get_ball_speed():
    return (global_trainer_data['ball_speed_x'], global_trainer_data['ball_speed_y'])

def get_all_players():
    return global_trainer_data['players'].copy()

def get_player_position(team_name, player_id):
    player_key = f"{team_name}_{player_id}"
    players = global_trainer_data['players']
    if player_key in players:
        return (players[player_key]['x'], players[player_key]['y'])
    return None

def get_player_speed(team_name, player_id):
    player_key = f"{team_name}_{player_id}"
    players = global_trainer_data['players']
    if player_key in players:
        return (players[player_key]['speed_x'], players[player_key]['speed_y'])
    return None

def get_player_orientation(team_name, player_id):
    player_key = f"{team_name}_{player_id}"
    players = global_trainer_data['players']
    if player_key in players:
        return (players[player_key]['body_angle'], players[player_key]['neck_angle'])
    return None


### ROBOT CLASS ###
class TritonClient:
    def __init__(self, teamname: str, side: str, id: int, orientations: tuple):
        self.teamname = teamname
        self.side = side
        self.id = id
        self.orientations = orientations
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.connect()


### MAIN FUNCTION ###
    def main(self):
        print("I am online")
        try: 
            self.send("(move -5 0)")
            time.sleep(15)

            for i in range(10):
                self.send("(dash 100)")
                time.sleep(0.1)
                print("YOOO IT WORKS " + str(get_player_position(self.teamname, self.id)))
            
            time.sleep(1)
            self.send("(kick 100 0)")
            time.sleep(1)

            for i in range(44):
                self.send("(dash 100)")
                time.sleep(0.1)
            
            time.sleep(2)
            self.send("(kick 50 0)")
            time.sleep(10)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.disconnect()
            print("done")


### BALL CHASING LOGIC ###
    def chase_ball(self):
        """Main ball chasing logic"""
        pass


### KICKING TOWARDS GOAL LOGIC ###
    def kick_towards_goal(self):
        """Kick the ball towards the goal"""
        pass

    def calculate_goal_angle(self):
        """Calculate the angle to the goal"""
        pass


### CORE FUNCTIONS ###
    def connect(self):
        self.sock.sendto(b"(init TritonBot (version 19))\0", UDP_CONFIG)
        (data, address) = self.sock.recvfrom(64)
        if m := re.search(INIT_PATTERN, data.decode()):
            self.side = m.group(1)
            self.id = int(m.group(2))
            self.addr = address
        else:
            raise Exception(f"Unexpected response: {data} from {address}")

    def disconnect(self):
        self.send("(bye)")

    def send(self, msg):
        try:
            msg_bytes = msg.encode() + b'\0'
            self.sock.sendto(msg_bytes, self.addr)
            print(f"Sent: {msg}")
        except Exception as e:
            print(f"Error sending command: {e}")

### RUN METHODS ###
def run_trainer():
    trainer_instance = trainer.Trainer()
    trainer_instance.set_global_data(global_trainer_data)
    trainer_instance.main()

def run_client():
    client = TritonClient("Triton", "left", 0, (0, 0, 0))
    client.main()

### MAIN ###
if __name__ == "__main__":
    client_process = multiprocessing.Process(target=run_client, name="TritonClient")
    trainer_process = multiprocessing.Process(target=run_trainer, name="Trainer")
    
    try:
        print("Starting trainer process...")
        trainer_process.start()
        print("Starting client process...")
        client_process.start()
        
        # Wait for both processes to complete
        client_process.join()
        trainer_process.join()
        
    except KeyboardInterrupt:
        print("Shutting down processes...")
        if client_process.is_alive():
            client_process.terminate()
        if trainer_process.is_alive():
            trainer_process.terminate()
        
        # Wait for processes to finish
        client_process.join()
        trainer_process.join()
        print("All processes terminated.")