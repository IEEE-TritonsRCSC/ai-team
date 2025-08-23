import re
import time
import random
import socket
import argparse
import sys
try:
    from TritonClient import TritonClient
except ImportError:
    print("Warning: TritonClient not found. Some features may not work.")
    TritonClient = None

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
            self.esp_sock.settimeout(1.0)  # 1 second timeout
            self.test_esp_connection()

        # Manual control variables
        self.manual_mode = False
        self.manual_commands = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'rotate_left': False,
            'rotate_right': False,
            'stop': False
        }

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

        # Run hardcoded test for a few cycles
        print("Running hardcoded test for 10 seconds...")
        test_start_time = time.time()
        while time.time() - test_start_time < 10:
            self.receive_information()
            self.execute_actions(self.get_actions())
            time.sleep(0.1)

        print("Hardcoded test finished. Switching to manual control mode...")
        print("\nQuick Commands: w/s=forward/back, a/d=strafe, q/e=rotate, x=stop")
        print("Custom Commands: kick <vx> <vy>, dash <speed>, turn <speed>")
        print("Wheel Control: wheel <1-4> <speed>")
        print("Type 'h' for full help, 'exit' to quit")

        # Switch to manual control mode
        self.manual_mode = True
        self.start_manual_control()

    def get_actions(self):
        actions = []
        for player in self.players[1::]:  # Skip dummy player
            action = player.get_action(self.ball_pos,
                                       self.poses[self.teamname])
            actions.append(action)
        return actions

    def execute_actions(self, actions):
        for player, action in zip(self.players[1::], actions):
            player.execute_action(action)
        if args.hardware:
            self.send_actions_to_esp(actions)

    def send_actions_to_esp(self, actions):
        """Send actions to ESP32 with error handling."""
        if not actions:
            return

        try:
            # Format as multi-line string with null terminator (TritonClient format)
            msg = "\n".join(actions) + "\0"
            self.esp_sock.sendto(msg.encode(), ESP_ADDR)
            print(f"Sent to ESP32: {actions}")

        except Exception as e:
            print(f"Error sending actions to ESP32: {e}")
            print("Check ESP32 connection and network settings")

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

    def test_esp_connection(self):
        """Test connection to ESP32."""
        print(f"Testing ESP32 connection at {ESP_IP}:{ESP_PORT}...")
        try:
            # Send test packet
            test_msg = "test"
            self.esp_sock.sendto(test_msg.encode(), ESP_ADDR)
            print(f"✓ Test packet sent to ESP32")

            # Check network configuration
            import subprocess
            result = subprocess.run(['ping', '-c', '1', ESP_IP],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print(f"✓ ESP32 is reachable via ping")
            else:
                print(f"⚠ ESP32 ping failed - check network connection")

        except Exception as e:
            print(f"⚠ ESP32 connection test failed: {e}")
            print("Continuing anyway - check ESP32 is powered and on correct network")

    def start_manual_control(self):
        """Start manual control mode with keyboard input."""
        print("\n=== MANUAL CONTROL MODE ACTIVE ===")
        print("Type commands and press Enter. Type 'h' for help.")

        try:
            while True:
                # Continue receiving information from server
                self.receive_information()

                # Get user input (blocking)
                try:
                    print("\nEnter command: ", end="", flush=True)
                    command = input().strip().lower()
                    if command:
                        self.process_manual_command(command)
                except EOFError:
                    break

                # Execute manual actions if hardware is enabled
                if args.hardware and self.manual_mode:
                    self.execute_manual_actions()

        except KeyboardInterrupt:
            print("\nExiting manual control mode...")
            if args.hardware:
                # Send stop command before exit
                self.esp_sock.sendto(b"bye", ESP_ADDR)
            sys.exit(0)

    def process_manual_command(self, command):
        """Process manual control commands."""
        if command == 'w':
            self.send_esp_command("dash 100")
            print("Moving forward")
        elif command == 's':
            self.send_esp_command("dash -100")
            print("Moving backward")
        elif command == 'a':
            self.send_esp_command("kick -100 0")
            print("Strafing left")
        elif command == 'd':
            self.send_esp_command("kick 100 0")
            print("Strafing right")
        elif command == 'q':
            self.send_esp_command("turn -50")
            print("Rotating left")
        elif command == 'e':
            self.send_esp_command("turn 50")
            print("Rotating right")
        elif command == 'x' or command == 'stop':
            self.send_esp_command("bye")
            print("Stopping all movement")
        elif command == 'm':
            self.manual_mode = not self.manual_mode
            print(f"Manual mode: {'ON' if self.manual_mode else 'OFF'}")
        elif command == 'h' or command == 'help':
            self.show_help()
        elif command.startswith('wheel'):
            self.handle_wheel_command(command)
        elif command == 'test':
            self.send_esp_command("test")
            print("Sending test command")
        elif command.startswith('kick'):
            self.handle_kick_command(command)
        elif command.startswith('dash'):
            self.handle_dash_command(command)
        elif command.startswith('turn'):
            self.handle_turn_command(command)
        elif command == 'exit' or command == 'quit':
            print("Exiting...")
            if args.hardware:
                self.send_esp_command("bye")
            sys.exit(0)
        else:
            print(f"Unknown command: {command}. Type 'h' for help.")

    def handle_wheel_command(self, command):
        """Handle individual wheel control commands."""
        # Format: wheel <1-4> <speed>
        parts = command.split()
        if len(parts) == 3:
            try:
                wheel_num = int(parts[1])
                speed = int(parts[2])
                if 1 <= wheel_num <= 4 and -1000 <= speed <= 1000:
                    # Send individual wheel command
                    wheel_cmd = f"wheel{wheel_num} {speed}"
                    self.send_esp_command(wheel_cmd)
                    print(f"Setting wheel {wheel_num} to speed {speed}")
                else:
                    print("Wheel number must be 1-4, speed must be -1000 to 1000")
            except ValueError:
                print("Invalid wheel command format. Use: wheel <1-4> <speed>")
        else:
            print("Invalid wheel command format. Use: wheel <1-4> <speed>")

    def handle_kick_command(self, command):
        """Handle kick command with custom parameters."""
        # Format: kick <vx> <vy>
        parts = command.split()
        if len(parts) == 3:
            try:
                vx = int(parts[1])
                vy = int(parts[2])
                kick_cmd = f"kick {vx} {vy}"
                self.send_esp_command(kick_cmd)
                print(f"Kick command: vx={vx}, vy={vy}")
            except ValueError:
                print("Invalid kick command. Use: kick <vx> <vy>")
        else:
            print("Invalid kick command format. Use: kick <vx> <vy>")

    def handle_dash_command(self, command):
        """Handle dash command with custom speed."""
        # Format: dash <speed>
        parts = command.split()
        if len(parts) == 2:
            try:
                speed = int(parts[1])
                dash_cmd = f"dash {speed}"
                self.send_esp_command(dash_cmd)
                print(f"Dash command: speed={speed}")
            except ValueError:
                print("Invalid dash command. Use: dash <speed>")
        else:
            print("Invalid dash command format. Use: dash <speed>")

    def handle_turn_command(self, command):
        """Handle turn command with custom speed."""
        # Format: turn <speed>
        parts = command.split()
        if len(parts) == 2:
            try:
                speed = int(parts[1])
                turn_cmd = f"turn {speed}"
                self.send_esp_command(turn_cmd)
                print(f"Turn command: speed={speed}")
            except ValueError:
                print("Invalid turn command. Use: turn <speed>")
        else:
            print("Invalid turn command format. Use: turn <speed>")

    def send_esp_command(self, command):
        """Send command to ESP32 if hardware is enabled."""
        if args.hardware:
            try:
                # Send command (manual format - single string)
                self.esp_sock.sendto(command.encode(), ESP_ADDR)
                print(f"✓ Sent to ESP32: {command}")

                # Optional: Add small delay to prevent overwhelming ESP32
                time.sleep(0.01)

            except socket.timeout:
                print(f"✗ Timeout sending command: {command}")
            except Exception as e:
                print(f"✗ Error sending to ESP32: {e}")
                print("Check ESP32 connection and network settings")
        else:
            print(f"Hardware not enabled. Would send: {command}")

    def execute_manual_actions(self):
        """Execute any pending manual actions."""
        # This can be used for continuous actions if needed
        pass

    def show_help(self):
        """Show help for manual control commands."""
        print("\n=== MANUAL CONTROL HELP ===")
        print("Quick Movement Commands:")
        print("  w - Move forward (dash 100)")
        print("  s - Move backward (dash -100)")
        print("  a - Strafe left (kick -100 0)")
        print("  d - Strafe right (kick 100 0)")
        print("  q - Rotate left (turn -50)")
        print("  e - Rotate right (turn 50)")
        print("  x/stop - Stop all movement")
        print("\nCustom Movement Commands:")
        print("  kick <vx> <vy> - Custom kick movement")
        print("    Example: kick 50 100")
        print("  dash <speed> - Custom forward/backward movement")
        print("    Example: dash 150")
        print("  turn <speed> - Custom rotation")
        print("    Example: turn -75")
        print("\nWheel Commands:")
        print("  wheel <1-4> <speed> - Control individual wheel")
        print("    Example: wheel 1 500")
        print("    Speed range: -1000 to 1000")
        print("    Wheel mapping: 1=FR, 2=BR, 3=BL, 4=FL")
        print("\nOther Commands:")
        print("  test - Send test command")
        print("  m - Toggle manual mode on/off")
        print("  h/help - Show this help")
        print("  exit/quit - Exit program")
        print("  ctrl+c - Force exit")
        print("\nNote: Hardware mode must be enabled (-hardware flag)")
        print("========================\n")


if __name__ == "__main__":
    args = parser.parse_args()
    trainer = Trainer("Triton", NUM_PLAYERS)
    trainer.main()
