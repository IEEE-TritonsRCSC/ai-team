import socket
import time
import math
import re
import threading

UDP_IP = "127.0.0.1"
UDP_PORT = 6000
UDP_CONFIG = (UDP_IP, UDP_PORT)

class TritonClient:
    def __init__(self, teamname: str, side: str, id: int, orientations: tuple):
        self.teamname = teamname
        self.side = side
        self.id = id
        self.orientations = orientations
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.1)  # Set timeout for non-blocking receive

        self.ball_distance = float('inf')
        self.ball_direction = 0
        self.player_x = 0
        self.player_y = 0
        self.goal_x = 52.5 if side == "left" else -52.5  # Goal position based on side
        self.goal_y = 0

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

### DATA PROCESSING LOGIC ###
    def start_listening(self):
        """Start a background thread to listen for incoming messages"""
        self.listen_thread = threading.Thread(target=self.receive_and_process, daemon=True)
        self.listen_thread.start()

    def receive_and_process(self):
        """Receive and process sensor data from the server"""
        while True:
            try:
                data, addr = self.sock.recvfrom(8192)
                message = data.decode('utf-8')
                print(message)
                
                # Parse vision data to find ball
                #if "(see" in message:
                if message.startswith('(see') or message.startswith('(hear') or message.startswith('(sense_body'):
                    self.parse_vision_data(message)
                    
            except socket.timeout:
                pass  # No data received, continue
            except Exception as e:
                print(f"Error receiving data: {e}")

    def parse_vision_data(self, message):
        """Parse vision data to extract ball and player positions"""
        try:
            # Extract ball information from vision data
            ball_match = re.search(r'\(\(ball\)\s+([\d\.\-]+)\s+([\d\.\-]+)(\s+([\d\.\-]+))?(\s+([\d\.\-]+))?\)', message)
            if ball_match:
                self.ball_distance = float(ball_match.group(1))
                self.ball_direction = float(ball_match.group(2))
                print(f"Ball: dist={self.ball_distance:.2f}, dir={self.ball_direction:.2f}")
            
            # Extract player position (if available)
            player_match = re.search(r'\(self\)\s+\(pol\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\)', message)
            if player_match:
                self.player_x = float(player_match.group(1))
                self.player_y = float(player_match.group(2))
                
        except Exception as e:
            print(f"Error parsing vision data: {e}")

### BALL CHASING LOGIC ###
    def chase_ball(self):
        """Main ball chasing logic"""
        if self.ball_distance == float('inf'):
            # No ball visible, search by turning
            self.send("(turn 45)")
            return
        
        # If ball is very close, try to kick it towards goal
        if self.ball_distance < 1.0:
            self.kick_towards_goal()
        else:
            # Move towards the ball
            self.move_towards_ball()

    def move_towards_ball(self):
        """Move towards the ball"""
        # Turn towards the ball
        if abs(self.ball_direction) > 5:  # If ball is not directly in front
            turn_angle = self.ball_direction
            # Limit turn angle to prevent excessive turning
            turn_angle = max(-45, min(45, turn_angle))
            self.send(f"(turn {turn_angle})")
        else:
            # Ball is in front, dash towards it
            dash_power = min(100, max(20, self.ball_distance * 10))  # Adjust power based on distance
            self.send(f"(dash {dash_power})")

### KICKING TOWARDS GOAL LOGIC ###
    def kick_towards_goal(self):
        """Kick the ball towards the goal"""
        # Calculate angle to goal
        goal_angle = self.calculate_goal_angle()
        
        # Turn towards goal if needed
        if abs(goal_angle) > 10:
            self.send(f"(turn {goal_angle})")
        else:
            # Kick towards goal
            kick_power = 100  # Strong kick
            self.send(f"(kick {kick_power} {goal_angle})")

    def calculate_goal_angle(self):
        """Calculate the angle to the goal"""
        # Simple calculation - in a real implementation you'd use trigonometry
        # For now, use the ball direction as approximation
        return self.ball_direction

### MAIN COMMANDS ###
    def connect(self):
        """Connect to the soccer server"""
        self.sock.bind((UDP_IP, 0))
        self.send("(init TritonBot (version 19))")
        print("Connected to server")

    def disconnect(self):
        """Disconnect from the soccer server"""
        self.send("(bye)")

    def send(self, msg: str):
        """Send a command to the server"""
        try:
            self.sock.sendto(msg.encode(), UDP_CONFIG)
            print(f"Sent: {msg}")
        except Exception as e:
            print(f"Error sending command: {e}")

# Create and run the client
client = TritonClient("Triton", "left", 0, (0, 0, 0))

if __name__ == "__main__":
    client.main()
    client.start_listening()
    client.receive_and_process()