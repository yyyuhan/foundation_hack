
from src.utils.comm_util import find_traj_frame_ids, get_device

# pytest -s tests/test_utils.py::test_get_device
def test_get_device():
    print(get_device())

# pytest -s tests/test_utils.py::test_find_traj_frame_ids
def test_find_traj_frame_ids():
    acc_map = {10: 0, 20: 1, 30: 2}
    traj_id, frame_id = find_traj_frame_ids(11, acc_map)
    assert traj_id == 1 and frame_id == 1
    traj_id, frame_id = find_traj_frame_ids(0, acc_map)
    assert traj_id == 0 and frame_id == 0
    traj_id, frame_id = find_traj_frame_ids(9, acc_map)
    assert traj_id == 0 and frame_id == 9
    traj_id, frame_id = find_traj_frame_ids(10, acc_map)
    assert traj_id == 1 and frame_id == 0
