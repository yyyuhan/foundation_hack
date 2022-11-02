
def get_device():
    from torch import cuda, device
    use_cuda = cuda.is_available()
    return device("cuda:0" if use_cuda else "cpu")

def find_traj_frame_ids(idx, acc_traj_map, sorted_keys=None):
    from bisect import bisect_left
    if sorted_keys is None:
        sorted_keys = sorted(acc_traj_map.keys())
    largest_smaller_key = bisect_left(sorted_keys, idx + 1)
    # import pdb
    # pdb.set_trace()
    key_idx = largest_smaller_key if largest_smaller_key else 0
    prev_frames = sorted_keys[largest_smaller_key - 1] if largest_smaller_key > 0 else 0
    traj_idx = acc_traj_map[sorted_keys[key_idx]]
    frame_idx = idx - prev_frames
    return traj_idx, frame_idx