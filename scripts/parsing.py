import argparse
import os
import pwd
from pytorch_lightning import Trainer
import itertools
import hashlib
import json
import copy
import inspect

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
POSS_VAL_NOT_LIST = (
    "Flag {} has an invalid list of values: {}. Length of list must be >=1"
)

REGISTRIES = {
    "LIGHTNING_REGISTRY": {},
    "DATASET_REGISTRY": {},
    "MODEL_REGISTRY": {},
    "LOSS_REGISTRY": {},
    "METRIC_REGISTRY": {},
    "OPTIMIZER_REGISTRY": {},
    "SCHEDULER_REGISTRY": {},
    "SEARCHER_REGISTRY": {},
    "CALLBACK_REGISTRY": {},
    "INPUT_LOADER_REGISTRY": {},
    "AUGMENTATION_REGISTRY": {},
    "LOGGER_REGISTRY": {},
}

INITED_OBJ = []

class GlobalNamespace(argparse.Namespace):
    pass

def parse_dispatcher_config(config):
    """
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
         *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    """

    assert all(
        [
            k
            in [
                "script",
                "available_gpus",
                "cartesian_hyperparams",
                "paired_hyperparams",
                "tune_hyperparams",
            ]
            for k in config.keys()
        ]
    )

    cartesian_hyperparamss = config["cartesian_hyperparams"]
    paired_hyperparams = config.get("paired_hyperparams", [])
    flags = []
    arguments = []
    experiment_axies = []

    # add anything outside search space as fixed
    fixed_args = ""
    for arg in config:
        if arg not in [
            "script",
            "cartesian_hyperparams",
            "paired_hyperparams",
            "available_gpus",
        ]:
            if type(config[arg]) is bool:
                if config[arg]:
                    fixed_args += "--{} ".format(str(arg))
                else:
                    continue
            else:
                fixed_args += "--{} {} ".format(arg, config[arg])

    # add paired combo of search space
    paired_args_list = [""]
    if len(paired_hyperparams) > 0:
        paired_args_list = []
        paired_keys = list(paired_hyperparams.keys())
        paired_vals = list(paired_hyperparams.values())
        flags.extend(paired_keys)
        for paired_combo in zip(*paired_vals):
            paired_args = ""
            for i, flg_value in enumerate(paired_combo):
                if type(flg_value) is bool:
                    if flg_value:
                        paired_args += "--{} ".format(str(paired_keys[i]))
                    else:
                        continue
                else:
                    paired_args += "--{} {} ".format(
                        str(paired_keys[i]), str(flg_value)
                    )
            paired_args_list.append(paired_args)

    # add every combo of search space
    product_flags = []
    for key, value in cartesian_hyperparamss.items():
        flags.append(key)
        product_flags.append(key)
        arguments.append(value)
        if len(value) > 1:
            experiment_axies.append(key)

    experiments = []
    exps_combs = list(itertools.product(*arguments))

    for tpl in exps_combs:
        exp = ""
        for idx, flg in enumerate(product_flags):
            if type(tpl[idx]) is bool:
                if tpl[idx]:
                    exp += "--{} ".format(str(flg))
                else:
                    continue
            else:
                exp += "--{} {} ".format(str(flg), str(tpl[idx]))
        exp += fixed_args
        for paired_args in paired_args_list:
            experiments.append(exp + paired_args)

    return experiments, flags, experiment_axies

def get_parser():
    global_namespace = GlobalNamespace(allow_abbrev=False)

    parser = argparse.ArgumentParser(
        description="Nox Standard Args.", allow_abbrev=False
    )

    # -------------------------------------
    # Run Setup
    # -------------------------------------
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Whether or not to train model",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Whether or not to run model on dev set",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Whether or not to run model on test set",
    )
    
    # -------------------------------------
    # System
    # -------------------------------------
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for each data loader [default: 4]",
    )

    # cache
    parser.add_argument(
        "--cache_path", type=str, default=None, help="Dir to cache images."
    )

    # -------------------------------------
    # Add object-level args
    # -------------------------------------

    def add_class_args(args_as_dict, parser):
        # for loop
        for argname, argval in args_as_dict.items():
            args_for_noxs = {
                a.dest: a for a in parser._actions if hasattr(a, "is_nox_action")
            }
            old_args = vars(parser.parse_known_args()[0])
            if argname in args_for_noxs:
                args_for_noxs[argname].add_args(parser, argval)
                newargs = vars(parser.parse_known_args()[0])
                newargs = {k: v for k, v in newargs.items() if k not in old_args}
                add_class_args(newargs, parser)

    parser.parse_known_args(namespace=global_namespace)
    add_class_args(vars(global_namespace), parser)

    return parser

def parse_args(args_strings=None):
    # run
    # Lightning 2.0 removes add_argparse_args
    parser = get_parser()
    sig = inspect.signature(Trainer.__init__)
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            parser.add_argument(f"--{name}")
        else:
            # Note that this does not store types, because many defaults are None
            parser.add_argument(f"--{name}", default=param.default)

    # legacy
    if not hasattr(Trainer, "add_argparse_args"):
        parser.add_argument(
            "--gpus",
            default=None,
            help="Number of GPUs to train on",
        )

    # Dataset
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--reward_fn_name", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ground_truth_column", type=str, required=True)

    # Optional with defaults
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--sync_ref_model", action="store_true")
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--results_path", type=str, default="results")

    # Deepspeed
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Max number of saved checkpoints")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="Comma-separated list of CUDA devices to use")
    parser.add_argument("--experiment_name", type=str, default="default_experiment", help="Name of the experiment")
    parser.add_argument("--local_rank", type=int, default=0, help="Used for distributed training.")


    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)

    args.devices = "auto"
        
    # using gpus
    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (
        isinstance(args.gpus, int) and args.gpus > 1
    ):
        args.strategy = "ddp"
        # should not matter since we set our sampler, in 2.0 this is default True and used to be called replace_sampler_ddp
        args.use_distributed_sampler = False
    else:
        if not hasattr(Trainer, "add_argparse_args"):  # lightning 2.0
            # args.strategy = "auto"
            args.strategy = "ddp"  # should be overwritten later in main
        else:  # legacy
            args.strategy = None
            args.replace_sampler_ddp = False

    # username
    args.unix_username = pwd.getpwuid(os.getuid())[0]

    # learning initial state
    args.step_indx = 1

    # set args
    args_for_noxs = {a.dest: a for a in parser._actions if hasattr(a, "is_nox_action")}
    for argname, argval in vars(args).items():
        if argname in args_for_noxs:
            args_for_noxs[argname].set_args(args, argval)
    '''
    # parse augmentations
    args.train_rawinput_augmentations = parse_augmentations(
        args.train_rawinput_augmentation_names
    )
    args.train_tnsr_augmentations = parse_augmentations(
        args.train_tnsr_augmentation_names
    )
    args.test_rawinput_augmentations = parse_augmentations(
        args.test_rawinput_augmentation_names
    )
    args.test_tnsr_augmentations = parse_augmentations(
        args.test_tnsr_augmentation_names
    )
    '''

    # parse tune parameters
    # args = parse_tune_params(args)

    return args

def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()