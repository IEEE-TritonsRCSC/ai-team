import time
from soccer_client import TritonClient

# UDP_IP = "127.0.0.1"
# UDP_PORT = 
# UDP_CONFIG = (UDP_IP, UDP_PORT)

if __name__ == "__main__":
    client = TritonClient("TritonBot", "l", 1, (0, 0, 0))
    try:
        time.sleep(5)
        client.move(-10, 10)

        while True:
            if client.received_message and client.received_message.startswith("(fullstate"):
                print("self position:", client.get_pos(client.received_message))
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.disconnect()