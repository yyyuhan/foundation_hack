from src.pipeline.load_atari import MspacmanDataProvider, MspacmanDataset
from src.utils.types import GameName

# TODO del
DATA_DIR = "/home/t-yuhancao/dev/foundation_hack/dataset/atari_v1"

# pytest -s tests/test_dataloader.py::test_dataset
def test_dataset():
    dataset = MspacmanDataset(DATA_DIR)
    state, action = dataset.__getitem__(1)
    # print(type(state))
    assert type(action) == int
    assert (state.shape) == (3, 210, 160)


# pytest -s tests/test_dataloader.py::test_dataloader
def test_dataloader():
    batch_size = 128
    dataloader = MspacmanDataProvider(DATA_DIR, GameName.ATARI_MSPACMAN, batch_size=batch_size)
    state_batch, action_batch = dataloader.get_batch()
    assert state_batch.shape == (batch_size, 3, 210, 160)
    assert action_batch.shape == (batch_size,)

# pytest -s tests/test_dataloader.py::test_data_transform
def test_data_transform():
    batch_size = 128
    dataloader = MspacmanDataProvider(DATA_DIR, GameName.ATARI_MSPACMAN, batch_size=batch_size)
    state_batch, action_batch = dataloader.get_batch()
    state_batch = [state for state in state_batch[:]]
    assert type(state_batch) == list
    assert state_batch[0].shape == (3, 210, 160)
