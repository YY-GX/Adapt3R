import h5py
import os

data_dir = "./data/libero/libero_90_processed"

missing_rgb_real = []

for filename in sorted(os.listdir(data_dir)):
    if not filename.endswith(".hdf5"):
        continue

    file_path = os.path.join(data_dir, filename)
    with h5py.File(file_path, 'r') as f:
        if 'data' not in f:
            continue

        demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]
        for demo_key in demo_keys:
            demo_path = f"data/{demo_key}/obs/robot0_eye_in_hand_rgb_real"
            try:
                obs_group = f['data'][demo_key]['obs']
                if 'robot0_eye_in_hand_rgb_real' not in obs_group:
                    missing_rgb_real.append(f"{filename} :: {demo_key}")
            except KeyError:
                missing_rgb_real.append(f"{filename} :: {demo_key} (no obs group)")

# Print only missing cases
if missing_rgb_real:
    print("\n🔍 Demos missing 'obs/robot0_eye_in_hand_rgb_real':")
    for entry in missing_rgb_real:
        print(f" - {entry}")
else:
    print("✅ All demos contain 'obs/robot0_eye_in_hand_rgb_real'")

