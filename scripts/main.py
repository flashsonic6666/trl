import sys
import os
import torch
import git
import time
import pickle
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.strategies import DDPStrategy
from parsing import parse_args
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from trl import GRPOConfig
from trl.trainer.grpo_vlm_trainer import VLMGRPOTrainer
from trl.reward_and_prompt_utils import get_reward_fn, get_prompt_template
import resource
import inspect

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
torch.multiprocessing.set_sharing_strategy("file_system")

os.environ["TORCH_EXTENSIONS_DIR"] = "/data/scratch/richwang/torch_extensions"


def train(args):
    """
    Train the model using the parsed arguments.
    """
    args.strategy = "auto"
    
    trainer_arg_names = inspect.signature(pl.Trainer.__init__).parameters.keys()
    trainer_kwargs = {k: v for k, v in vars(args).items() if k in trainer_arg_names}
    trainer = pl.Trainer(**trainer_kwargs)

    
    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object
    log.info(
        "\nProject running by author: {} \ndate:{}, \nfrom commit: {} -- {}".format(
            commit.author,
            time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(commit.committed_date)),
            commit.hexsha,
            commit.message,
        )
    )
    
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files=args.dataset_file)
    processor = AutoProcessor.from_pretrained(args.model_name)
    reward_fn = get_reward_fn(args.reward_fn_name)
    prompt_template_fn = get_prompt_template(args.prompt_template)
    
    model = AutoModelForVision2Seq.from_pretrained(args.model_name)
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=10,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        use_vllm=args.use_vllm,
        sync_ref_model=args.sync_ref_model,
        num_generations=args.num_generations,
        deepspeed="./deepspeed_config_zero3.json",
        bf16=True,
        gradient_checkpointing=True,
        save_total_limit=1,
    )
    
    trainer = VLMGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset['train'],
        prompt_template_fn=prompt_template_fn,
        ground_truth_column=args.ground_truth_column,
    )
    
    trainer.train()
    
    if args.local_rank == 0:
        print("Saving args to {}.args".format(args.results_path))
        pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))
    
    return model, trainer

if __name__ == "__main__":
    args = parse_args()
    model, trainer = train(args)
