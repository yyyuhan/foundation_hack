def get_device():
    from torch import cuda, device

    use_cuda = cuda.is_available()
    return device("cuda:0" if use_cuda else "cpu")


def find_traj_frame_ids(idx, acc_traj_map, sorted_keys=None):
    from bisect import bisect_left

    if sorted_keys is None:
        sorted_keys = sorted(acc_traj_map.keys())
    largest_smaller_key = bisect_left(sorted_keys, idx + 1)
    key_idx = largest_smaller_key if largest_smaller_key else 0
    prev_frames = sorted_keys[largest_smaller_key - 1] if largest_smaller_key > 0 else 0
    traj_idx = acc_traj_map[sorted_keys[key_idx]]
    frame_idx = idx - prev_frames
    return traj_idx, frame_idx


def save_checkpoint(model, name, path_to_save="checkpoints"):
    from torch import save

    raw_model = model.module if hasattr(model, "module") else model
    print(f"===saving checkpoint to {path_to_save}")
    save(raw_model.state_dict(), f"{path_to_save}/{name}.pt")


# def load_checkpoint(model, model_path):
#     from torch import load
#     import torch.nn as nn

#     model_path = model_path + ".pt"
#     print(type(load(model_path)))
#     model.load_state_dict(state_dict=load(model_path))
