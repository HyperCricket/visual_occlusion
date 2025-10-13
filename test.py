"""Gripper interaction demo.

This script illustrates the process of importing grippers into a scene and making it interact
with the objects with actuators. It also shows how to procedurally generate a scene with the
APIs of the MJCF utility functions.

Example:
    $ python run_gripper_test.py
"""

import xml.etree.ElementTree as ET
import numpy as np

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.grippers import PandaGripper, RethinkGripper
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_actuator, new_joint

# Updated imports for modern robosuite
import mujoco
import mujoco.viewer

if __name__ == "__main__":

    # start with an empty world
    world = MujocoWorldBase()

    # add a table
    arena = TableArena(table_full_size=(0.4, 0.4, 0.05), table_offset=(0, 0, 1.1), has_legs=False)
    world.merge(arena)

    # add a gripper
    gripper = RethinkGripper()
    # Create another body with a slider joint to which we'll add this gripper
    gripper_body = ET.Element("body", name="gripper_base")
    gripper_body.set("pos", "0 0 1.3")
    gripper_body.set("quat", "0 0 1 0")  # flip z
    gripper_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 1", damping="50"))
    # Add the dummy body with the joint to the global worldbody
    world.worldbody.append(gripper_body)
    # Merge the actual gripper as a child of the dummy body
    world.merge(gripper, merge_body="gripper_base")
    # Create a new actuator to control our slider joint
    world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

    # add an object for grasping
    mujoco_object = BoxObject(
        name="box", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001]
    ).get_obj()
    # Set the position of this object
    mujoco_object.set("pos", "0 0 1.11")
    # Add our object to the world body
    world.worldbody.append(mujoco_object)

    # add reference objects for x and y axes
    x_ref = BoxObject(
        name="x_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual", joints=None
    ).get_obj()
    x_ref.set("pos", "0.2 0 1.105")
    world.worldbody.append(x_ref)
    y_ref = BoxObject(
        name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1], obj_type="visual", joints=None
    ).get_obj()
    y_ref.set("pos", "0 0.2 1.105")
    world.worldbody.append(y_ref)

    # start simulation with modern mujoco
    xml_string = world.get_xml()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # for gravity correction
    gravity_corrected = ["gripper_z_joint"]
    _ref_joint_vel_indexes = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in gravity_corrected]

    # Set gripper parameters
    gripper_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_z")
    gripper_z_low = 0.07
    gripper_z_high = -0.02
    gripper_z_is_low = False

    gripper_jaw_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in gripper.actuators]
    gripper_open = [-0.0115, 0.0115]
    gripper_closed = [0.020833, -0.020833]
    gripper_is_closed = True

    # hardcode sequence for gripper looping trajectory
    seq = [(False, False), (True, False), (True, True), (False, True)]

    step = 0
    T = 500
    
    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if step % 100 == 0:
                print("step: {}".format(step))

                # Get contact information
                for i in range(data.ncon):
                    contact = data.contact[i]
                    geom_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                    if geom_name1 == "floor" and geom_name2 == "floor":
                        continue

                    print("geom1: {}, geom2: {}".format(geom_name1, geom_name2))
                    print("friction: {}".format(contact.friction))
                    print("normal: {}".format(contact.frame[0:3]))

            # Iterate through gripping trajectory
            if step % T == 0:
                plan = seq[int(step / T) % len(seq)]
                gripper_z_is_low, gripper_is_closed = plan
                print("changing plan: gripper low: {}, gripper closed {}".format(gripper_z_is_low, gripper_is_closed))

            # Control gripper
            if gripper_z_is_low:
                data.ctrl[gripper_z_id] = gripper_z_low
            else:
                data.ctrl[gripper_z_id] = gripper_z_high
            if gripper_is_closed:
                data.ctrl[gripper_jaw_ids] = gripper_closed
            else:
                data.ctrl[gripper_jaw_ids] = gripper_open

            # Step through sim
            mujoco.mj_step(model, data)
            # Apply gravity compensation
            for idx in _ref_joint_vel_indexes:
                qvel_addr = model.jnt_qposadr[idx]
                data.qfrc_applied[qvel_addr] = data.qfrc_bias[qvel_addr]
            
            viewer.sync()
            step += 1
