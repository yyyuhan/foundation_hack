# part of the implementation was modified from https://github.com/yobibyte/atarigrandchallenge/blob/master/agc/dataset.py
from torch.utils.data import Dataset, DataLoader
from src.utils.types import ErrMsg
# import agc.dataset as ds
# import agc.util as util
from os import path, listdir
import cv2
from src.utils.comm_util import find_traj_frame_ids

def get_env_dataset(env_name):
    dataset_map = {"mspacman": MspacmanDataset}
    if env_name not in dataset_map:
        raise ValueError(f"{ErrMsg.InvalidParam}: {env_name}")
    return dataset_map[env_name]    

class AtariDataset(Dataset):
    TRAJS_SUBDIR = 'trajectories'
    SCREENS_SUBDIR = 'screens'

    def __init__(self, data_path, game):
        super().__init__()
        self.trajs_path = path.join(data_path, AtariDataset.TRAJS_SUBDIR, game)       
        self.screens_path = path.join(data_path, AtariDataset.SCREENS_SUBDIR, game)

        assert path.exists(self.trajs_path), f"{ErrMsg.InvalidParam}: traj path {self.trajs_path} does not exist"
        assert path.exists(self.screens_path), f"{ErrMsg.InvalidParam}: screen path {self.screens_path} does not exist"

        self.total_frames = None
        self.total_trajs = None

    def load_trajectories(self):
        trajectories = {}
        acc_traj_map = {}
        total_frames = 0
        for traj in listdir(self.trajs_path):
            curr_traj = []
            with open(path.join(self.trajs_path, traj)) as f:
                for i,line in enumerate(f):
                    #first line is the metadata, second is the header
                    if i > 1:
                        curr_data = line.rstrip('\n').replace(" ","").split(',')
                        curr_trans = {}
                        curr_trans['frame']    = int(curr_data[0])
                        curr_trans['reward']   = int(curr_data[1])
                        curr_trans['score']    = int(curr_data[2])
                        curr_trans['terminal'] = int(curr_data[3])
                        curr_trans['action']   = int(curr_data[4])
                        curr_traj.append(curr_trans)  
            traj_id = int(traj.split('.txt')[0])
            trajectories[traj_id] = curr_traj
            total_frames += len(curr_traj)
            acc_traj_map[total_frames] = traj_id
        
        # update global info
        self.total_trajs = len(trajectories.keys())
        self.total_frames = total_frames
        # sum([len(self.trajectories[traj]) for traj in self.trajectories])
        self.acc_traj_map = acc_traj_map
        self.acc_traj_sorted_keys = sorted(acc_traj_map.keys()) # ready for bsearch

        self.traj_map = trajectories
    
    @property
    def total_frames(self):
        if self.total_frames is None:
            raise RuntimeError(f"{ErrMsg.InitFailure}: total_frames not initialized")
        else:
            return self.total_frames

    @property
    def total_trajs(self):
        if self.total_trajs is None:
            raise RuntimeError(f"{ErrMsg.InitFailure}: total_trajs not initialized")
        else:
            return self.total_trajs
        

class MspacmanDataset(AtariDataset):
    GAME = 'mspacman'
    def __init__(self, data_path, num_trajs=0) -> None:
        super().__init__(data_path, self.GAME)
        self.load_trajectories()
        
    def __len__(self):
        return len(self.total_frames)

    def __getitem__(self, idx):
        # load frames on the run
        traj_idx, frame_idx = find_traj_frame_ids(idx, self.acc_traj_map, self.acc_traj_sorted_keys)
        # load picture
        frame_path = path.join(self.screens_path, traj_idx, f"{frame_idx}.png")
        state = cv2.imread(frame_path)
        action = self.traj_map[traj_idx][frame_idx]['action']
        # state = preprocess(cv2.imread(path.join(st_dir, str(pid) + '.png'), cv2.IMREAD_GRAYSCALE))

        return state, action

class MspacmanDataProvider:
    def __init__(self, env_name, batch_size) -> None:
        dataset = get_env_dataset(env_name)()
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_batch(self):
        return next(iter(self.data_loader))