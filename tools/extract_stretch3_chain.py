"""Print the Stretch 3 kinematic chain constants used by stretchwbc.chain.

The numbers come from MuJoCo Menagerie's hello_robot_stretch_3/stretch.xml:
for each body on the path from base_link to link_grasp_center, its fixed
offset from its parent (position, quaternion) and its joint (type, axis,
position). Run it after updating Menagerie and paste the output into
chain.py; tests/test_chain_mujoco.py then checks the two still agree.

    python tools/extract_stretch3_chain.py path/to/stretch.xml
"""

import sys

import mujoco
import numpy as np

CHAIN = [
    "link_lift",
    "link_arm_l4",
    "link_arm_l3",
    "link_arm_l2",
    "link_arm_l1",
    "link_arm_l0",
    "link_wrist_yaw",
    "link_DW3_wrist_pitch",
    "link_SG3_gripper_body",
    "link_grasp_center",
]


def main(path: str) -> None:
    model = mujoco.MjModel.from_xml_path(path)
    print(
        "# name, body position, body quaternion (w, x, y, z), joint (name, type, axis, position)"
    )
    print("CHAIN = (")
    for name in CHAIN:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = np.round(model.body_pos[body], 10).tolist()
        quat = np.round(model.body_quat[body], 10).tolist()
        joint = None
        if model.body_jntnum[body] == 1:
            j = model.body_jntadr[body]
            kind = {mujoco.mjtJoint.mjJNT_SLIDE: "slide", mujoco.mjtJoint.mjJNT_HINGE: "hinge"}[
                model.jnt_type[j]
            ]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            joint = (
                joint_name,
                kind,
                np.round(model.jnt_axis[j], 10).tolist(),
                np.round(model.jnt_pos[j], 10).tolist(),
            )
        print(f"    ({name!r}, {pos}, {quat}, {joint!r}),")
    print(")")


if __name__ == "__main__":
    main(sys.argv[1])
