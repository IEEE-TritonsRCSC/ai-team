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
    def __init__(self, teamname: str, side: str, id: int, orientations: tuple):
        self.teamname = teamname
        self.side = side
        self.id = id
        self.orientations = orientations
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = True
        self.last_action = None
        self.kicked_off = False  # Track if the client has kicked off
        self.next_action = None  # Store next manual action to execute
        self.received_message = None
        self.connect()
        self.start_listening()

    def connect(self):
        init_message = f"(init {self.teamname})"
        self.sock.sendto(init_message.encode(), UDP_CONFIG)
        print(f"Sent: {init_message}")

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
                if message.startswith('(see') or message.startswith('(hear') or message.startswith('(sense_body'):
                    if '(hear 0 referee kick_off_l)' in message:
                        self.kicked_off = True
                    self.received_message = message  # Store the received message
                    self.send_cycle_action()
                    
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
            # Default behavior: simple movement pattern
            if self.last_action:
                action_type, args = self.last_action
            else:
                action_type, args = 'turn', [0]

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
        message = f"(turn {angle})"
        self.sock.sendto(message.encode(), UDP_CONFIG)
        print(f"Sent: {message}")

    def _send_dash(self, power):
        message = f"(dash {power})"
        self.sock.sendto(message.encode(), UDP_CONFIG)
        print(f"Sent: {message}")

    def _send_kick(self, power, angle=0):
        message = f"(kick {power} {angle})"
        self.sock.sendto(message.encode(), UDP_CONFIG)
        print(f"Sent: {message}")

    def _send_catch(self, angle=0):
        message = f"(catch {angle})"
        self.sock.sendto(message.encode(), UDP_CONFIG)
        print(f"Sent: {message}")

    def _send_move(self, x, y):
        message = f"(move {x} {y})"
        self.sock.sendto(message.encode(), UDP_CONFIG)
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

    def chase_ball(self, message):
        ball_regex = r'\(\(ball\)\s+([\d\.\-]+)\s+([\d\.\-]+)(\s+([\d\.\-]+))?(\s+([\d\.\-]+))?\)'
        match = re.search(ball_regex, message)
        if match:
            distance = float(match.group(1))  # 0.7
            angle = float(match.group(2))      # 64
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
        message = "(bye)"
        self.sock.sendto(message.encode(), UDP_CONFIG)
        print(f"Sent: {message}")
        time.sleep(0.1)  # Give time for the message to be sent
        self.sock.close()