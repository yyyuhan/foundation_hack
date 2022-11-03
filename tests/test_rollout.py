from src.env.atari_env import AtariEnv
from src.training.trainer import Trainer
from src.utils.comm_util import load_checkpoint


# pytest -s tests/test_rollout.py::test_env
def test_env():
    env = AtariEnv("")
    model = load_checkpoint(model_path="checkpoints/bc_750")
    trainer = Trainer(model=model)
    state = env.reset()
    for s in 10:
        action = trainer.pred_action(state)
        state, _, _, _ = env.step(action)
