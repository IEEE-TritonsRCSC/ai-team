import socket
import time
import threading
import re
from constants.client_constants import *

UDP_IP = "127.0.0.1"
UDP_PORT = 6000
UDP_CONFIG = (UDP_IP, UDP_PORT)

# UDP client to send a message to the simulation server
class TritonClient:
    def __init__(self, teamname: str, side: str, id: int, fullstate: bool = True):
        self.teamname = teamname
        self.side = side
        self.id = id
        self.fullstate = fullstate
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = True
        self.last_action = None
        self.kicked_off = False  # Track if the client has kicked off
        self.next_action = None  # Store next manual action to execute
        self.received_message = None
        self.server_addr = UDP_CONFIG  # Initialize with default, will be updated after connect
        self.connect()
        self.start_listening()

    def connect(self):
        init_message = f"(init {self.teamname} (version 19))\0"
        self.sock.sendto(init_message.encode(), UDP_CONFIG)
        print(f"Sent: {init_message}")
        
        # Wait for server response to get the assigned port and confirm registration
        try:
            data, addr = self.sock.recvfrom(8192)
            message = data.decode('utf-8', errors='ignore')
            print(f"Init response: {message}")
            
            # Parse the init response to extract player info
            # Expected format: (init l 1 before_kick_off) or (init r 2 before_kick_off)
            init_regex = r'\(init ([lr]) (\d+) ([^)]+)\)'
            match = re.search(init_regex, message)
            
            if match:
                confirmed_side = match.group(1)
                assigned_id = int(match.group(2))
                game_mode = match.group(3)
                
                print(f"Registration confirmed:")
                print(f"  Side: {confirmed_side}")
                print(f"  Player ID: {assigned_id}")
                print(f"  Game mode: {game_mode}")
                
                # Update player ID if it was assigned by server
                self.id = assigned_id
                self.side = confirmed_side
                
                # Update the server address to use the port from the response
                self.server_addr = addr
                print(f"Server assigned address: {self.server_addr}")
                
                return True
            else:
                print(f"Error: Could not parse init response: {message}")
                return False
                
        except Exception as e:
            print(f"Error during connection: {e}")
            return False

    def start_listening(self):
        """Start a background thread to listen for incoming messages"""
        self.listen_thread = threading.Thread(target=self.listen_for_messages, daemon=True)
        self.listen_thread.start()

    def listen_for_messages(self):
        """Continuously listen for messages from the server"""
        self.sock.settimeout(1.0)  # Set timeout to avoid blocking indefinitely
        while self.running:
            try:
                data, addr = self.sock.recvfrom(8192)  # Buffer size of 8192 bytes
                message = data.decode('utf-8', errors='ignore')
                print(f"Received: {message}")
                
                # Check if this is a sensory message that requires a response
                if '(hear 0 referee kick_off_l)' in message:
                    self.kicked_off = True
                self.received_message = message  # Store the received message
                self.send_cycle_action()
                # time.sleep(0.1)
                    
            except socket.timeout:
                continue  # Continue listening if timeout occurs
            except Exception as e:
                if self.running:  # Only print error if we're still supposed to be running
                    print(f"Error receiving message: {e}")
                break

    def send_cycle_action(self):
        """Send an action command in response to sensory information"""
        # Check if there's a manual action queued
        if self.next_action:
            action_type, args = self.next_action
            self.last_action = self.next_action
            self.next_action = None  # Clear the queued action
        else:
            if self.last_action and not self.kicked_off:
                action_type, args = self.last_action
            elif self.kicked_off:
                action_type, args = 'turn', [0]
            else:
                return

        if action_type == 'turn':
            self._send_turn(args[0])
        elif action_type == 'dash':
            self._send_dash(args[0])
        elif action_type == 'kick':
            self._send_kick(args[0], args[1] if len(args) > 1 else 0)
        elif action_type == 'catch':
            self._send_catch(args[0] if len(args) > 0 else 0)
        elif action_type == 'move':
            self._send_move(args[0], args[1])

    # Internal methods that actually send the commands
    def _send_turn(self, angle):
        message = f"(turn {angle})\0"
        self.sock.sendto(message.encode(), self.server_addr)
        print(f"Sent: {message}")

    def _send_dash(self, power):
        message = f"(dash {power})\0"
        self.sock.sendto(message.encode(), self.server_addr)
        print(f"Sent: {message}")

    def _send_kick(self, power, angle=0):
        message = f"(kick {power} {angle})\0"
        self.sock.sendto(message.encode(), self.server_addr)
        print(f"Sent: {message}")

    def _send_catch(self, angle=0):
        message = f"(catch {angle})\0"
        self.sock.sendto(message.encode(), self.server_addr)
        print(f"Sent: {message}")

    def _send_move(self, x, y):
        message = f"(move {x} {y})\0"
        self.sock.sendto(message.encode(), self.server_addr)
        print(f"Sent: {message}")

    def turn(self, angle):
        """Queue a turn action to be executed in the next cycle"""
        assert CLIENT_ANGLE[0] <= angle <= CLIENT_ANGLE[1], "Angle out of range"
        self.next_action = ('turn', [angle])
        print(f"Queued: turn {angle}")

    def dash(self, power, angle=0):
        """Queue a dash action to be executed in the next cycle"""
        assert CLIENT_DASH_POWER[0] <= power <= CLIENT_DASH_POWER[1], "Power out of range"
        if angle != 0:
            assert CLIENT_ANGLE[0] <= angle <= CLIENT_ANGLE[1], "Angle out of range"
        self.next_action = ('dash', [power, angle])
        print(f"Queued: dash {power} {angle}")

    def kick(self, power, angle=0):
        """Queue a kick action to be executed in the next cycle"""
        assert CLIENT_DASH_POWER[0] <= power <= CLIENT_DASH_POWER[1], "Power out of range"
        self.next_action = ('kick', [power, angle])
        print(f"Queued: kick {power} {angle}")

    def catch(self, angle=0):
        """Queue a catch action to be executed in the next cycle"""
        self.next_action = ('catch', [angle])
        print(f"Queued: catch {angle}")

    def move(self, x, y):
        """Queue a move action to be executed in the next cycle"""
        self.next_action = ('move', [x, y])
        print(f"Queued: move {x} {y}")

    def get_pos(self, message):
        if self.fullstate:
            pos_regex = rf'\(\(p {self.side} {self.id} \d\)\s+([\d\.\-]+)\s+([\d\.\-]+)'
            match = re.search(pos_regex, message)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                return (x, y)
            else:
                return None
        else:
            raise NotImplementedError("Position extraction not implemented for non-fullstate mode")
            
    def chase_ball(self, message):
        ball_regex = r'\(\(ball\)\s+([\d\.\-]+)\s+([\d\.\-]+)(\s+([\d\.\-]+))?(\s+([\d\.\-]+))?\)'
        match = re.search(ball_regex, message)
        if match:
            distance = float(match.group(1))
            angle = float(match.group(2))
            print(distance, angle)
            if abs(distance) < KICKABLE_MARGIN:
                return True
            if abs(angle) >= 3.0:
                self.turn(angle)
                return False
            self.dash(min(50, distance * 10))
            return False
        else:
            return False

    def disconnect(self):
        self.running = False  # Stop the listening thread
        message = "(bye)\0"
        self.sock.sendto(message.encode(), self.server_addr)
        print(f"Sent: {message}")
        time.sleep(0.1)  # Give time for the message to be sent
        self.sock.close()