from src.models.beit import ImageBeit
from src.utils.cmd_util import common_arg_parser, parse_cmdline_kwargs
from src.utils.types import ErrMsg
from src.training.trainer import Trainer

import torch
import math

from src.utils.comm_util import get_device

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
        model = ImageBeit(num_actions=0) # TODO dataset
        model = model.to(device=comm_args.device)
        # prepare optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=comm_args.learning_rate, weight_decay=comm_args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_warmup_decay)
        # get trainer
        trainer = Trainer(model=model, optimizer=optimizer, lr_scheduler=scheduler)
        trainer.train(n_iters=comm_args.max_iters)

    elif "eval" == task:
        pass
    else:
        raise ValueError(f"{ErrMsg.TypeNotSupported}: {task}")