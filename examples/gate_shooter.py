import time
import numpy as np
from soccer_client import TritonClient
from constants.field_constants import GOAL_L, GOAL_R
from constants.client_constants import KICKABLE_MARGIN

if __name__ == "__main__":
    client1 = TritonClient("TritonBot", "l", 1, (0, 0, 0), print_messages=True)
    client2 = TritonClient("TritonBot", "l", 2, (0, 0, 0), print_messages=True)
    try:
        client1.move(-5, -20)
        client2.move(-10, 10)

        while not client1.kicked_off:
            time.sleep(0.1)

        goal = GOAL_R if client1.side == 'l' else GOAL_L
        current_kicker = client2

        while True:
            if current_kicker == client1:
                current_kicker = client2
                while True:
                    ball_pose = client2.state.get('ball_pose', None)
                    if not ball_pose:
                        time.sleep(0.1)
                        continue
                    vec = np.array(ball_pose[:2]) - np.array(goal)
                    destination = np.array(goal) + vec + (vec / np.linalg.norm(vec)) * KICKABLE_MARGIN / 2
                    destination_theta = np.degrees(np.arctan2(vec[1], vec[0]))
                    if client2.goto(destination[0], destination[1], margin=KICKABLE_MARGIN / 2, theta=destination_theta, speed=50):
                        print("Got ball!")
                        pos = client2.state['self_pose']
                        ball_pos = client2.state['ball_pose']
                        print(f"Self position: {pos}")
                        print(f"Ball position: {ball_pos}")
                        angle_to_goal = np.degrees(np.arctan2(goal[1] - pos[1], goal[0] - pos[0]))
                        print(f"Angle ball to gate: {angle_to_goal:.2f}")
                        angle_diff = (angle_to_goal - pos[4] + 180) % 360 - 180
                        client2.kick(80, angle_diff)
                        print("Kicked!")
                        break
            else:
                current_kicker = client1
                while True:
                    ball_pose = client1.state.get('ball_pose', None)
                    if not ball_pose:
                        time.sleep(0.1)
                        continue
                    vec = np.array(ball_pose[:2]) - np.array(goal)
                    destination = np.array(goal) + vec + (vec / np.linalg.norm(vec)) * KICKABLE_MARGIN * 2/3
                    destination_theta = np.degrees(np.arctan2(vec[1], vec[0]))
                    if client1.goto(destination[0], destination[1], margin=KICKABLE_MARGIN / 3, theta=destination_theta, speed=50):
                        print("Got ball!")
                        pos = client1.state['self_pose']
                        ball_pos = client1.state['ball_pose']
                        print(f"Self position: {pos}")
                        print(f"Ball position: {ball_pos}")
                        angle_to_goal = np.degrees(np.arctan2(goal[1] - pos[1], goal[0] - pos[0]))
                        print(f"Angle ball to gate: {angle_to_goal:.2f}")
                        angle_diff = (angle_to_goal - pos[4] + 180) % 360 - 180
                        client1.kick(50, angle_diff)
                        print("Kicked!")
                        break
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client1.disconnect()
        client2.disconnect()