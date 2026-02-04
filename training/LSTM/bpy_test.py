import bpy
import numpy as np

# --- Load generated motion ---
# Y_pred_np shape: (1, seq_len, 33*3)
generated = np.load(r"C:\Users\IBRA\project_maham\models\LSTM\Y_pred_pn.npy")  # or save Y_pred_np

seq_len = generated.shape[1]

# --- Map joint indices to your Blender armature bones ---
# Replace these names with your actual Blender rig bone names
bone_names = [
    "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    # ... continue for all 33 joints
]

armature = bpy.data.objects["Armature"]  # your armature object name
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

# --- Insert keyframes ---
for f in range(seq_len):
    frame_num = f + 1  # Blender frame index
    bpy.context.scene.frame_set(frame_num)
    
    for j, bone_name in enumerate(bone_names):
        bone = armature.pose.bones[bone_name]
        
        # Extract xyz for this joint
        x, y, z = generated[0, f, j*3:(j+1)*3]
        
        # Set bone location relative to armature (adjust if needed)
        bone.location = (x, y, z)
        
        # Insert keyframe for location
        bone.keyframe_insert(data_path="location", frame=frame_num)

print("Animation inserted successfully.")
