import h5py
import numpy as np
import os

# Process ONLY this file
filename = "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5"
data_dir = "/mnt/arc/yygx/pkgs_baselines/Adapt3R/data/libero/libero_90_processed"
file_path = os.path.join(data_dir, filename)

print(f"\n🔄 Processing: {filename}")

try:
    with h5py.File(file_path, 'a') as f:
        if 'data' not in f:
            print("  ⚠️ Skipped: no 'data' group found.")
        else:
            demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]

            for demo_key in demo_keys:
                demo_group = f['data'][demo_key]
                if 'obs' not in demo_group:
                    print(f"  ⚠️ Skipped {demo_key}: no 'obs' group")
                    continue

                obs_group = demo_group['obs']

                # Only process demos missing rgb_real
                if 'robot0_eye_in_hand_rgb_real' in obs_group:
                    print(f"  ⏩ Skipped {demo_key}: already has rgb_real")
                    continue

                # Rename original RGB to rgb_real if it exists
                if 'robot0_eye_in_hand_rgb' in obs_group:
                    obs_group.move('robot0_eye_in_hand_rgb', 'robot0_eye_in_hand_rgb_real')
                    print(f"  ✅ Renamed: {demo_key}/obs/robot0_eye_in_hand_rgb → robot0_eye_in_hand_rgb_real")
                else:
                    print(f"  ⚠️ Skipped rename: {demo_key} (no RGB to rename)")

                # Duplicate depth to RGB
                if 'robot0_eye_in_hand_depth' in obs_group and 'robot0_eye_in_hand_rgb' not in obs_group:
                    depth = obs_group['robot0_eye_in_hand_depth'][:]  # shape (T, H, W, 1)
                    depth_rgb = np.repeat(depth, 3, axis=-1)           # shape (T, H, W, 3)

                    obs_group.create_dataset(
                        'robot0_eye_in_hand_rgb', data=depth_rgb, compression="gzip"
                    )
                    print(f"  ✅ Created fake RGB from depth in {demo_key}")
                elif 'robot0_eye_in_hand_rgb' in obs_group:
                    print(f"  ⏩ Fake RGB already exists in {demo_key}")
                else:
                    print(f"  ⚠️ Skipped fake RGB: {demo_key} (no depth)")
except Exception as e:
    print(f"  ❌ ERROR processing {filename}: {e}")

