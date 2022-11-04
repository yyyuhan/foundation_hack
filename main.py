from src.env.atari_env import AtariEnv
from tests.test_dataloader import DATA_DIR
from src.pipeline.load_atari import MspacmanDataProvider
from src.models.beit import ImageBeit
from src.utils.cmd_util import common_arg_parser, parse_cmdline_kwargs
from src.utils.types import ErrMsg, GameName, GameSet
from src.training.trainer import Trainer

import torch
import math

from src.utils.comm_util import gen_traj_video, get_device

def parse_cmd_args():
    arg_parser = common_arg_parser()
    args_ns, extra_args_list = arg_parser.parse_known_args()
    extra_args_dict = parse_cmdline_kwargs(extra_args_list)
    return args_ns, extra_args_dict


if __name__ == "__main__":
    # parse cmd-line args
    comm_args, extra_args = parse_cmd_args()

    device = get_device()
    task = comm_args.task
    warmup_steps = comm_args.warmup_steps
    total_steps = comm_args.num_steps_per_iter * comm_args.max_iters
    batch_size = comm_args.batch_size
    checkpoint = comm_args.model
    action_space_dim = GameSet.action_space_dim(game=GameName.ATARI_MSPACMAN)

    def lr_warmup_decay(steps):
        if steps < warmup_steps:
            # linear warmup
            lr_mult = float(steps) / float(warmup_steps)
        else:
            # cosine learning rate decay
            progress = float(steps - warmup_steps) / float(total_steps - warmup_steps)
            lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr_mult

    if "train" == task:
        # prepare model
        model = ImageBeit(num_actions=action_space_dim, device=device)
        # prepare optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=comm_args.learning_rate, weight_decay=comm_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_warmup_decay)
        # get data loader
        dataloader = MspacmanDataProvider(DATA_DIR, GameName.ATARI_MSPACMAN, batch_size=batch_size)

        # get trainer
        trainer = Trainer(model=model, batch_size=batch_size, optimizer=optimizer, lr_scheduler=scheduler, training_data_provider=dataloader, device=device)
        trainer.train(n_iters=comm_args.max_iters)

    elif "rollout" == task:
        if not checkpoint:
            raise ValueError(f"invalid checkpoint param for rollout task")
        # load model
        model = ImageBeit(num_actions=action_space_dim, device=device)
        model.load_checkpoint(model_path=comm_args.model)
        trainer = Trainer(model=model, device=device)
        env = AtariEnv(game=GameName.ATARI_MSPACMAN)
        state, _ = env.reset()
        frame_list = []
        traj_no = 0
        for s in range(total_steps):
            frame_list.append(state)
            action = trainer.pred_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                gen_traj_video("rollouts", traj_no, frame_list)
                frame_list.clear()
                traj_no += 1
                state, _ = env.reset()

    elif "eval" == task:
        pass
    else:
        raise ValueError(f"{ErrMsg.TypeNotSupported}: {task}")