import numpy as np

keypoint_path = r"C:\Users\IBRA\project_maham\data\keypoints\walk1_motion.npy"
k_tensor = np.load(keypoint_path)
print(np.shape(k_tensor))

def recenter(tensor) :
    T, num_joints, num_coords = tensor.shape
    for t in range(T):
        left_hip = tensor[t, 23, :3]
        right_hip = tensor[t, 24, :3]
        root= (left_hip + right_hip) / 2
        for j in range(num_joints):
            tensor[t, j, :3] -=root
    return tensor


def create_window(local_motion, local_vel, root_vel, window = 60, stride = 10):
    X_motion = []
    X_vel = []
    X_root = []
    Y = []
    T = local_motion.shape[0]
    for start in range(0, T - window - 1, stride):
        end = start +window
        x_motion = local_motion[start:end]
        x_vel = local_vel[start:end]
        x_root = root_vel[start:end]
        
        y = local_motion[start+1:end+1]
        
        X_motion.append(x_motion)
        X_vel.append(x_vel)
        X_root.append(x_root)
        Y.append(y)
    return(np.array(X_motion), np.array(X_vel), np.array(X_root), np.array(Y))


T, J, C = k_tensor.shape
root_motion = np.zeros((T, 3), dtype=np.float32)

for t in range(T):
    left_hip = k_tensor[t, 23, :3]
    right_hip = k_tensor[t, 24, :3]
    root_motion[t] = (left_hip + right_hip) / 2

recentered_tensor = recenter(k_tensor.copy())
local_motion = np.copy(recentered_tensor)
local_motion = local_motion[:, :, :3]
root_vel = root_motion[1:] - root_motion[:-1]
local_vel = local_motion[1:, :, :3] - local_motion[:-1, :, :3]

X_motion, X_vel, X_root, Y = create_window(
    local_motion,
    local_vel,
    root_vel,
    window=60,
    stride=10
)

print("X_motion:", X_motion.shape)
print("X_vel:", X_vel.shape)
print("X_root:", X_root.shape)
print("Y:", Y.shape)

print("root vel shape --->",np.shape(root_vel))
print("local vel shape --->",np.shape(local_vel))
print("root vel sam --->",root_vel[0])
print("local vel sam --->",local_vel[0][0])
print(root_motion[0][:])
print(local_motion[0][0][:])
print(np.shape(root_motion))
print(k_tensor[0][0][:])
print(recentered_tensor[0][0][:])
print(recentered_tensor[0][23][:])
print(recentered_tensor[0][24][:])


# np.save(r"C:\Users\IBRA\project_maham\data\processed\walk1_root_vel.npy", root_vel)
# np.save(r"C:\Users\IBRA\project_maham\data\processed\walk1_local_vel.npy", local_vel)