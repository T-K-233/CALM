import numpy as np
from isaacgym import gymapi
from utils.motion_lib import MotionLib

# data = np.load("calm/data/motions/amp_humanoid_walk.npy", allow_pickle=True)

# new_data = data.tolist()


# print(data.tolist()["rotation"])

# # joint rotations in (x, y, z, w) order

# new_data["rotation"]["arr"][:, 0:15, 0] = 0
# new_data["rotation"]["arr"][:, 0:15, 1] = 0
# new_data["rotation"]["arr"][:, 0:15, 2] = 0
# new_data["rotation"]["arr"][:, 0:15, 3] = 1

# # new_data["root_translation"]["arr"][:, :] = 0

# new_data = np.array(new_data)

# print(new_data.tolist()["rotation"])

# np.save("calm/data/motions/amp_humanoid_what.npy", new_data, allow_pickle=True)



_dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
_dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
_key_body_ids = [ 5,  8, 11, 14]

device = "cpu"

mlib = MotionLib(
    motion_file="calm/data/motions/amp_humanoid_walk.npy",
    dof_body_ids=_dof_body_ids,
    dof_offsets=_dof_offsets,
    key_body_ids=_key_body_ids,
    equal_motion_weights=False,
    device=device
)

print("length:", mlib.get_motion_length([0]))

motion_sequence = []

for t in range(1, 1300, 33):
    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = mlib.get_motion_state([0], t * 0.001)
    motion_sequence.append(dof_pos.to("cpu").detach().numpy())

motion_sequence = np.array(motion_sequence)

np.save("dof_profile.npy", motion_sequence)
np.save("/home/tk/Desktop/CS285-Final-Proj/dof_profile.npy", motion_sequence)
