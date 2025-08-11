import time
from soccer_client import TritonClient

if __name__ == "__main__":
    client = TritonClient("TritonBot", "l", 1, (0, 0, 0))
    try:
        client.move(-10, 0)

        while True:
            if client.received_message and client.received_message.startswith("(fullstate"):
                print("Got full state!")
                break
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.disconnect()