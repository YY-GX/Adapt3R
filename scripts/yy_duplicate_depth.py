import h5py
import numpy as np
import os

data_dir = "/mnt/arc/yygx/pkgs_baselines/Adapt3R/data/libero/libero_90_processed"

for filename in os.listdir(data_dir):
    if not filename.endswith(".hdf5"):
        continue

    file_path = os.path.join(data_dir, filename)
    print(f"\n🔄 Processing: {filename}")

    with h5py.File(file_path, 'a') as f:
        demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]

        for demo_key in demo_keys:
            obs_group = f['data'][demo_key]['obs']

            # Rename RGB to rgb_real if it exists and hasn't already been renamed
            if 'robot0_eye_in_hand_rgb' in obs_group and 'robot0_eye_in_hand_rgb_real' not in obs_group:
                obs_group.move('robot0_eye_in_hand_rgb', 'robot0_eye_in_hand_rgb_real')
                print(f"  ✅ Renamed: {demo_key}/obs/robot0_eye_in_hand_rgb → robot0_eye_in_hand_rgb_real")
            elif 'robot0_eye_in_hand_rgb_real' in obs_group:
                print(f"  ⏩ Already renamed: {demo_key}")
            else:
                print(f"  ⚠️ Skipped rename: {demo_key} (no rgb)")

            # Create new fake RGB from depth
            if 'robot0_eye_in_hand_depth' in obs_group and 'robot0_eye_in_hand_rgb' not in obs_group:
                depth = obs_group['robot0_eye_in_hand_depth'][:]  # (T, H, W, 1)
                depth_rgb = np.repeat(depth, 3, axis=-1)           # (T, H, W, 3)

                obs_group.create_dataset(
                    'robot0_eye_in_hand_rgb', data=depth_rgb, compression="gzip"
                )
                print(f"  ✅ Created fake RGB from depth in {demo_key}")
            elif 'robot0_eye_in_hand_rgb' in obs_group:
                print(f"  ⏩ Fake RGB already exists in {demo_key}")
            else:
                print(f"  ⚠️ Skipped fake RGB: {demo_key} (no depth)")
