import h5py
import os

file_path = "/mnt/arc/yygx/pkgs_baselines/Adapt3R/data/libero/libero_90_processed/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5"

with h5py.File(file_path, 'r') as f:
    for demo_key in f['data'].keys():
        if not demo_key.startswith('demo_'):
            continue
        demo_path = f['data'][demo_key]
        if 'obs' in demo_path and 'robot0_eye_in_hand_rgb_real' in demo_path['obs']:
            rgb_real = demo_path['obs']['robot0_eye_in_hand_rgb_real']
            print(f"{demo_key} → {rgb_real.shape}")
        else:
            print(f"{demo_key} → ❌ MISSING rgb_real")

