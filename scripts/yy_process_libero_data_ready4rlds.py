import os
import h5py
import hydra
import numpy as np
from natsort import natsorted
from libero.libero import benchmark
from tqdm import trange
import adapt3r.utils.pytorch3d_transforms as pt
import torch
import adapt3r.env.libero.utils as lu
from hydra.utils import instantiate
import robosuite.utils.transform_utils as T


compress_keys = {
    'robot0_eye_in_hand_pointcloud_full',
    'agentview_pointcloud_full'
}

# ✅ Cleaned + renamed keys only
def dump_demo(demo, file_path, demo_i, attrs=None):
    with h5py.File(file_path, 'a') as f:
        group_data = f['data']
        group = group_data.create_group(demo_i)

        if attrs is not None:
            for key in attrs:
                group.attrs[key] = attrs[key]

        group.attrs['num_samples'] = len(demo['actions'])

        non_obs_keys = ('actions', 'abs_actions', 'terminated', 'truncated', 'reward', 'success')
        group.create_dataset('states', data=())

        for key in demo:
            if key in non_obs_keys:
                group.create_dataset(key, data=demo[key])
            else:
                if key in compress_keys:
                    data = np.array(demo[key]).astype(np.float16)
                else:
                    data = demo[key]
                group.create_dataset(f'obs/{key}', data=data)


def process_demo(old_demo, env):
    """
    Collect gt depths in simulation by replaying demos and produce final observation keys.
    """

    actions = old_demo['actions']
    states = old_demo['states']
    init_state = states[0]

    obs, info = env.reset(init_state)

    new_demo = {
        'actions': [],
        'abs_actions': [],
        'reward': [],
        'terminated': [],
        'truncated': [],
        'success': [],
        'ee_states': [],
        'gripper_states': [],
        'joint_states': [],
        'eye_in_hand_rgb': [],
        'eye_in_hand_rgb_real': [],
    }

    # Keep agentview and other visual obs (non-robot0)
    for key in obs:
        if not key.startswith('robot0_'):
            new_demo[key] = []

    for t in trange(len(actions), disable=True):
        # Keep only non-robot0 obs directly
        for key in new_demo:
            if key in obs and key not in ('eye_in_hand_rgb', 'eye_in_hand_rgb_real'):
                new_demo[key].append(obs[key])

        # Add renamed obs
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']  # shape (4,)
        eef_rot = T.quat2axisangle(eef_quat)  # shape (3,)
        new_demo['ee_states'].append(np.concatenate([eef_pos, eef_rot], axis=-1))  # shape (6,)
        new_demo['gripper_states'].append(obs['robot0_gripper_qpos'])
        new_demo['joint_states'].append(obs['robot0_joint_pos'])

        # Original RGB → real
        new_demo['eye_in_hand_rgb_real'].append(obs['robot0_eye_in_hand_rgb'])

        # Depth → fake RGB
        depth = obs['robot0_eye_in_hand_depth']
        new_demo['eye_in_hand_rgb'].append(np.repeat(depth, 3, axis=-1))

        obs, reward, terminated, truncated, info = env.step(actions[t])

        controller = env.env.robots[0].controller
        goal_pos = controller.goal_pos
        goal_ori = pt.matrix_to_rotation_6d(torch.tensor(controller.goal_ori)).numpy()
        abs_action = np.concatenate((goal_pos, goal_ori, actions[t][..., -1:]))

        new_demo['actions'].append(actions[t])
        new_demo['abs_actions'].append(abs_action)
        new_demo['reward'].append(reward)
        new_demo['terminated'].append(terminated)
        new_demo['truncated'].append(truncated)
        new_demo['success'].append(info['success'])

    return new_demo


def process_task_dataset(task_no, source_h5_path, dest_h5_path, benchmark, env_factory):
    """
    Generate preprocessed data for a task.
    """
    demos = h5py.File(source_h5_path, 'r')['data']
    demo_keys = natsorted(list(demos.keys()))

    with h5py.File(dest_h5_path, 'a') as f:
        already_processed = 0
        if 'data' not in f:
            group_data = f.create_group('data')
            for attr in demos.attrs:
                group_data.attrs[attr] = demos.attrs[attr]
        else:
            already_processed = len(set(f['data']))

    env = env_factory(task_id=task_no, benchmark=benchmark, img_height=256, img_width=256)

    for idx in trange(already_processed, len(demo_keys), disable=0):
        demo_k = demo_keys[idx]
        demo = process_demo(demos[demo_k], env)
        dump_demo(demo, dest_h5_path, demo_k)


@hydra.main(config_path="../config", config_name='collect_data', version_base=None)
def main(cfg):
    is_debug = True

    source_dir = os.path.join(cfg.data_prefix, cfg.task.suite_name, cfg.task.benchmark_name + '_unprocessed')
    save_dir = os.path.join(cfg.data_prefix, cfg.task.suite_name, cfg.task.benchmark_name + '_processed')
    os.makedirs(save_dir, exist_ok=True)

    print(source_dir)
    print(save_dir)

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[cfg.task.benchmark_name]()
    env_factory = instantiate(cfg.task.env_factory)

    if is_debug:
        debug_file = "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5"
        print(f"[DEBUG MODE] Only processing: {debug_file}")
        source_h5_path = os.path.join(source_dir, debug_file)
        dest_h5_path = os.path.join(save_dir, debug_file)
        process_task_dataset(
            task_no=0,
            source_h5_path=source_h5_path,
            dest_h5_path=dest_h5_path,
            benchmark=benchmark_instance,
            env_factory=env_factory
        )
        return

    task_names = benchmark_instance.get_task_names()
    task_files = [os.path.join(source_dir, benchmark_instance.get_task_demonstration(i).split('/')[1])
                  for i in range(benchmark_instance.get_num_tasks())]

    task_nums = cfg.task_nums if 'task_nums' in cfg else None

    for task_no, task_name in enumerate(task_names):
        if task_nums is not None and task_no not in task_nums:
            continue

        setting, number, _ = lu.deconstruct_task_name(task_name)

        if 'setting_filter' in cfg and setting != cfg.setting_filter:
            continue
        if 'late_start' in cfg and task_no < cfg.late_start:
            continue

        print(task_name)
        source_h5_path = task_files[task_no]
        dest_h5_path = os.path.join(save_dir, f"{task_name}_demo.hdf5")
        process_task_dataset(task_no, source_h5_path, dest_h5_path, benchmark_instance, env_factory)


if __name__ == "__main__":
    main()
