import time
from soccer_client import TritonClient

if __name__ == "__main__":
    client = TritonClient("TritonBot", "l", 1, (0, 0, 0))
    try:
        client.move(-10, 0)

        while not client.kicked_off:
            time.sleep(0.1)

        while True:
            if client.received_message and client.chase_ball(client.received_message):
                print("Got ball!")
                client.kick(100)
                break
            time.sleep(0.1)
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.disconnect()