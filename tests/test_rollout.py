from src.env.atari_env import AtariEnv
from src.models.beit import ImageBeit
from src.training.trainer import Trainer
from src.utils.comm_util import get_device
from src.utils.types import GameName


# pytest -s tests/test_rollout.py::test_env
def test_env():
    device = get_device()
    env = AtariEnv(game=GameName.ATARI_MSPACMAN)
    model = ImageBeit(num_actions=18, device=device)
    model.load_checkpoint(model_path="checkpoints/bc_750.pt")
    trainer = Trainer(model=model, device=device)
    state, _ = env.reset()
    for s in range(10):
        action = trainer.pred_action(state)
        state, _, _, _, _ = env.step(action)


def test_rollout():
    env = AtariEnv("")
