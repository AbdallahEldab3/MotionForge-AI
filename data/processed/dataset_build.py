import numpy as np
import os
keypoint_dir = r"C:\Users\IBRA\project_maham\data\keypoints\new_data"
files = [f for f in os.listdir(keypoint_dir) if f.endswith(".npy")]
files.sort()

def recenter(tensor) :
    T, num_joints, num_coords = tensor.shape
    for t in range(T):
        left_hip = tensor[t, 23, :3]
        right_hip = tensor[t, 24, :3]
        root= (left_hip + right_hip) / 2
        for j in range(num_joints):
            tensor[t, j, :3] -=root
    return tensor

local_motions = []
local_vels = []
root_vels = []

for f in files :
    path = os.path.join(keypoint_dir, f)
    k_tensor = np.load(path)[:, :, :3]
    
    
    T, J, C = k_tensor.shape
    root_motion = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        left_hip = k_tensor[t, 23, :3]
        right_hip = k_tensor[t, 24, :3]
        root_motion[t] = (left_hip + right_hip) / 2
        
    recentered = recenter(k_tensor.copy())
    local_motion = np.copy(recentered)
    root_vel = root_motion[1:] - root_motion[:-1]
    local_vel = local_motion[1:, :, :] - local_motion[:-1, :, :]
    local_motion = local_motion[1:]
    
    local_motions.append(local_motion)
    local_vels.append(local_vel)
    root_vels.append(root_vel)

local_motions = np.concatenate(local_motions, axis = 0)
local_vels = np.concatenate(local_vels, axis = 0)
root_vels = np.concatenate(root_vels, axis = 0)
    
def create_window(local_motion, local_vel, root_vel, window=60, stride=10):
    X_motion, X_vel, X_root, Y = [], [], [], []
    T = local_motion.shape[0]

    for start in range(0, T - window - 1, stride):
        end = start + window
        X_motion.append(local_motion[start:end])
        X_vel.append(local_vel[start:end])
        X_root.append(root_vel[start:end])
        Y.append(local_motion[start+1:end+1])

    return np.array(X_motion), np.array(X_vel), np.array(X_root), np.array(Y)

X_motion, X_vel, X_root, Y = create_window(local_motions, local_vels, root_vels, window=60, stride=10)

T, J, C = k_tensor.shape
root_motion = np.zeros((T, 3), dtype=np.float32)

N, T, J, C = X_motion.shape
X_motion_flat = X_motion.reshape(N, T, J * C)
X_vel_flat = X_vel.reshape(N, T, J * C)
Y_flat = Y.reshape(N, T, J * C)

X_input = np.concatenate(
    [X_motion_flat, X_vel_flat, X_root],
    axis=-1
)

X_2d = X_input.reshape(-1, X_input.shape[-1])
Y_2d = Y_flat.reshape(-1, Y_flat.shape[-1])

eps = 1e-8

X_mean = X_2d.mean(axis = 0)
X_std = X_2d.std(axis= 0) + eps

Y_mean = Y_2d.mean(axis= 0)
Y_std = Y_2d.std(axis= 0) + eps

X_norm = (X_input - X_mean) / X_std
Y_norm = (Y_flat - Y_mean) / Y_std

np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_norm.npy", X_norm)
np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_norm.npy", Y_norm)

np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_mean.npy", X_mean)
np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\X_std.npy",  X_std)
np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_mean.npy", Y_mean)
np.save(r"C:\Users\IBRA\project_maham\data\keypoints\new_data\Y_std.npy",  Y_std)


