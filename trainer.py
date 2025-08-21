import socket
import re

UDP_IP = "127.0.0.1"
UDP_PORT = 6001
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
            self.addr = address
        
        self.cycle = 0
        self.goals = {'left': {'x': -52.5, 'y': 0}, 'right': {'x': -52.5, 'y': 0}}
        self.ball = {'x': 0, 'y': 0, 'speed_x': 0, 'speed_y': 0}
        self.players = {}
        
        # Global data reference
        self.global_data = None

    def set_global_data(self, global_data):
        """Set global data reference for inter-process communication"""
        self.global_data = global_data

    ### PARSING LOGIC ###
    def parse_trainer_vision(self, data):
        try:
            message = data.decode('utf-8').strip()
            
            if not message.startswith('(see_global'):
                return
            
            # Extract cycle number
            cycle_match = re.search(r'\(see_global (\d+)', message)
            if cycle_match:
                self.cycle = int(cycle_match.group(1))
                # Update global data
                if self.global_data:
                    self.global_data['cycle'] = self.cycle
            
            # Parse ball information
            ball_match = re.search(r'\(\(b\)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\)', message)
            if ball_match:
                self.ball = {
                    'x': float(ball_match.group(1)),
                    'y': float(ball_match.group(2)),
                    'speed_x': float(ball_match.group(3)),
                    'speed_y': float(ball_match.group(4))
                }
                # Update global data
                if self.global_data:
                    self.global_data['ball_x'] = self.ball['x']
                    self.global_data['ball_y'] = self.ball['y']
                    self.global_data['ball_speed_x'] = self.ball['speed_x']
                    self.global_data['ball_speed_y'] = self.ball['speed_y']
            
            # Parse players' information
            player_matches = re.findall(r'\(\(p\s+"([^"]+)"\s+(\d+)\)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\)', message)
            
            for match in player_matches:
                team_name, player_id, x, y, speed_x, speed_y, body_angle, neck_angle = match
                player_key = f"{team_name}_{player_id}"
                self.players[player_key] = {
                    'team': team_name,
                    'id': int(player_id),
                    'x': float(x),
                    'y': float(y),
                    'speed_x': float(speed_x),
                    'speed_y': float(speed_y),
                    'body_angle': float(body_angle),
                    'neck_angle': float(neck_angle)
                }
            
            # Update global data with players
            if self.global_data:
                self.global_data['players'] = self.players.copy()
            
            self.print_game_state()
            
        except Exception as e:
            print(f"Error parsing global vision: {e}")

    ### PRINT LOGS ###
    def print_game_state(self):
        """Print a more compact version of the game state"""
        print(f"Cycle {self.cycle:3d} | Ball: ({self.ball['x']:6.1f}, {self.ball['y']:6.1f}) | "
              f"Speed: ({self.ball['speed_x']:6.2f}, {self.ball['speed_y']:6.2f}) | "
              f"Players: {len(self.players)}")
        
        # Show player positions briefly
        for player_key, player_data in self.players.items():
            print(f"  {player_key}: ({player_data['x']:6.1f}, {player_data['y']:6.1f}) "
                  f"body={player_data['body_angle']:6.1f}°")
    

    ### MAIN FUNCTION ###
    def main(self):
        self.sock.sendto(b"(eye on)\0", self.addr)
        while True:
            (data, address) = self.sock.recvfrom(1024)
            if address == self.addr:
                # Parse the global vision data
                self.parse_trainer_vision(data)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()