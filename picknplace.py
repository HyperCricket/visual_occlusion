import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper

# 1️⃣ Create environment
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=False,
    reward_shaping=True,
    control_freq=20,
    hard_reset=False,
)
env = VisualizationWrapper(env, indicator_configs=None)

# Reset environment
obs = env.reset()

# 2️⃣ Helper function to step with smooth interpolation
def move_to(env, target_action, steps=100):
    """Move smoothly to target action over N steps."""
    for i in range(steps):
        env.step(target_action)
        env.render()
        time.sleep(0.02)

# 3️⃣ Define control structure
# Action format: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
# gripper: +1 open, -1 close
def create_action(dx=0, dy=0, dz=0, gripper=0):
    return np.array([dx, dy, dz, 0, 0, 0, gripper])

# 4️⃣ Define pick-and-place sequence
pick_and_place_seq = [
    ("approach above cube", create_action(0, 0, 0.05, gripper=1)),
    ("lower to cube", create_action(0, 0, -0.05, gripper=1)),
    ("close gripper", create_action(0, 0, 0, gripper=-1)),
    ("lift cube", create_action(0, 0, 0.05, gripper=-1)),
    ("move right", create_action(0.1, 0, 0, gripper=-1)),
    ("lower to place", create_action(0, 0, -0.05, gripper=-1)),
    ("open gripper", create_action(0, 0, 0, gripper=1)),
    ("lift away", create_action(0, 0, 0.05, gripper=1)),
]

# 5️⃣ Execute multiple cycles
for cycle in range(3):  # run 3 pick-and-place cycles
    print(f"\n=== Starting cycle {cycle+1} ===")
    for desc, action in pick_and_place_seq:
        print(f"Action: {desc}")
        move_to(env, action, steps=100)

print("✅ Completed all pick-and-place cycles")
env.close()
