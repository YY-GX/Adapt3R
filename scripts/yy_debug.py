import h5py

file_path = "./data/libero/libero_90_processed/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5"

def print_keys(name, obj):
    if isinstance(obj, h5py.Dataset):
        shape_str = f" | {obj.shape}"
    else:
        shape_str = ""
    print(f"{name}{shape_str}")

with h5py.File(file_path, 'r') as f:
    demo_keys = [key for key in f['data'].keys() if key.startswith('demo_')]
    for demo_key in demo_keys:
        print(f"\n--- {demo_key} ---")
        f['data'][demo_key].visititems(
            lambda name, obj: print_keys(f"data/{demo_key}/{name}", obj)
        )

        if 'actions' in f['data'][demo_key]:
            length = f['data'][demo_key]['actions'].shape[0]
            print(f"Length: {length} steps")
        else:
            print("Length: N/A (no 'actions' dataset)")
