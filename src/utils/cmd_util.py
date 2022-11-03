def arg_parser():
    """Create an empty argparse.ArgumentParser."""
    import argparse

    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """Create an argparse.ArgumentParser for common cmd-line args."""
    parser = arg_parser()
    parser.add_argument("--task", help="task", type=str, choices=["train", "rollout"], default="train")
    parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("-d", "--daemon", dest="daemon", action="store_true", help="run in daemon mode")
    # parser.add_argument("--conf_file", help="config file name", type=str)
    parser.add_argument("--log_path", help="directory to save logs", type=str, default=None, required=False)
    # parser.add_argument("--prev_model", help="continue training from a previous trained model", type=str, default="")
    parser.add_argument("--max_epsteps", help="maximum steps for each episode", type=int, default=None, required=False)
    parser.add_argument("--warmup_steps", help="warmup steps", type=int, default=10000, required=False)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4, required=False)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--num_steps_per_iter", type=int, default=300, required=False)

    return parser


def specific_arg_parser(args):
    """Create an argparse.ArgumentParser for specific cmd-line args."""
    assert isinstance(args, dict), f'{"Input parameters for specific_arg_parser must be dictionary"}'
    parser = arg_parser()
    for key in args:
        parser.add_argument(f"--{key}", type=type(args[key]), default=None)

    return parser


def parse_unknown_args(args):
    """Parse arguments not consumed by arg parser into a dictionary

    Args: args (list): Unkown args from cmd line. e.g. 2nd value returned by argparse.ArgumentParser.parse_known_args
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith("--"):
            if "=" in arg:
                key = arg.split("=")[0][2:]
                value = arg.split("=")[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def parse_cmdline_kwargs(args):
    """Convert cmd-line arguments to a dict"""

    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}
