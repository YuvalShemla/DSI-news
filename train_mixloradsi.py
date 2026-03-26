import copy
import os
from pathlib import Path
from pprint import pprint

import msgspec
import torch
from safetensors import safe_open
from t5_pretrainer.dataset import RiporForSeq2seqCollator, RiporForSeq2seqDataset
from t5_pretrainer.mixlora_config import MixLoraConfig
from t5_pretrainer.mixlora_trainer import MixLoraDSI_Varigrow_Trainer
from t5_pretrainer.mixlora_varigrow import MixLoraDSI_Varigrow
from t5_pretrainer.ripor import RiporForSeq2seq
from t5_pretrainer.utils.utils import is_first_worker, set_seed
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.utils import logging

torch.set_printoptions(sci_mode=False)

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
MODEL_ARGS_NAME = "model_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

model_dict = {
    "ripor": RiporForSeq2seq,
    "mixloradsi": MixLoraDSI_Varigrow,
}

set_seed(42)


def get_argument_and_config():
    parser = HfArgumentParser(Seq2SeqTrainingArguments)  # type: ignore
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--pretrained_path", default=None, type=str, required=True)
    parser.add_argument("--query_to_docid_path", default=None, type=str, required=True)
    parser.add_argument("--docid_to_smtid_path", default=None, type=str, required=True)
    parser.add_argument(
        "--mixlora_config_json_path",
        default="./MixLoraDSI/mixlora_config.json",
        type=str,
        required=True,
    )
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--taskid", type=int, default=0)

    # Only parse to HF's TrainingArguments valid keys
    training_args = parser.parse_args_into_dataclasses()[0]

    # Full arguments with added arguments
    args = parser.parse_args()

    # MixLoRA config
    with open(args.mixlora_config_json_path, "rb") as f:
        mixlora_config = msgspec.json.Decoder().decode(f.read())

    mixlora_config = MixLoraConfig.from_config(mixlora_config)

    return training_args, args, mixlora_config


def get_dataset_and_data_collator(args):
    # Dataset
    train_dataset = RiporForSeq2seqDataset(
        example_path=args.query_to_docid_path,
        docid_to_smtid_path=args.docid_to_smtid_path,
    )

    # Data collator
    data_collator = RiporForSeq2seqCollator(tokenizer_type="t5-base", max_length=256) ### !!! This should load from the trained tokenizer !!!???

    simple_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=False,
        collate_fn=data_collator,
    )

    return train_dataset, data_collator, simple_dataloader


def get_model(args, mixlora_config):
    model_cls = model_dict[args.model_name]
    if args.model_name == "ripor":
        model = model_cls.from_pretrained(
            model_name_or_path=args.pretrained_path,
            model_args=None,
        )
    else:
        model = model_cls.from_pretrained(
            model_name_or_path=args.pretrained_path,
            mixlora_config=mixlora_config,
        )

    checkpoint = {}
    if (Path(args.pretrained_path) / "model.safetensors").exists():
        with safe_open(Path(args.pretrained_path) / "model.safetensors", framework="pt", device="cpu") as f:  # type: ignore
            # This is most likely t5-self-neg checkpoint
            for k in f.keys():
                checkpoint[k] = f.get_tensor(k)
    elif (Path(args.pretrained_path) / "pytorch_model.bin").exists():
        checkpoint = torch.load(
            os.path.join(args.pretrained_path, "pytorch_model.bin"), map_location="cpu"
        )

    # For KL regularization
    if mixlora_config.kl_loss:
        model.previous_checkpoint = copy.deepcopy(model.base_model).cuda()

    return model, checkpoint


def main():
    training_args, args, mixlora_config = get_argument_and_config()

    train_dataset, data_collator, simple_dataloader = get_dataset_and_data_collator(args)

    # Model
    model, _ = get_model(args, mixlora_config)
    model.taskid = args.taskid

    
    if args.taskid == 1 and "-pt" in args.pretrained_path: 
        # NOTE!!!! THIS VARIGROW VERSION DOES NOT TRAIN ONE EPOCH ON D0, WHICH IS MORE REALISTIC. THE NOVELTY STATS ARE ACCUMULATED STARTING FROM D1.

        # We assume that the first task D1 will use 2 experts per layer, since MoE only works with 2+ experts.
        model._freeze_base_model(freeze_vocab=mixlora_config.freeze_vocab) # Since we do not run a before_task here
        trainer = MixLoraDSI_Varigrow_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            freeze_modules=None,
            mixlora_config=mixlora_config,
        )
    else:  # Continual learning model
        """
        Before training the model, we need to:
        1. Freeze the base model
        2. Extend the model with new experts and router weights
        """
        novelty_results, grad_mask_dict = model._before_task(
            train_loader=simple_dataloader, taskid=args.taskid, freeze_vocab=mixlora_config.freeze_vocab,
        )
        
        del simple_dataloader

        print(" ***************** Novelty results ***************** ")
        pprint(novelty_results)

        trainer = MixLoraDSI_Varigrow_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            freeze_modules=None,
            mixlora_config=mixlora_config,
            grad_mask_dict=grad_mask_dict,
        )

    print(" ***************** MixLoRA config ***************** ")
    pprint(model.mixlora_config)
    print(" ***************** MixLoRA config ***************** ")

    if is_first_worker():
        data_collator.tokenizer.save_pretrained(trainer.args.output_dir)
    os.makedirs(
        os.path.join("./MixLoraDSI/logs", args.run_name),
        exist_ok=True,
    )
    # Clear everything in the log directory
    os.system(f"rm -rf ./MixLoraDSI/logs/{args.run_name}/*")

    trainer.train()

    if args.taskid == 1:
        # Update the base_novelty stats if they are all zeros, which means they are not initialized from D0, but D1 instead.

        # First check if the base novelty stats of the first expert of the first layer is all zeros
        if torch.all(model.base_model.decoder_base_novelty_mean.state_dict()['layer0_expert0'] == 0.0):
            model.update_base_novelty_stats()

    trainer.save_torch_model_and_tokenizer(data_collator.tokenizer)
    # Move the model from checkpoint-* directory to the output directory
    os.system(
        f"mv {trainer.args.output_dir}/checkpoint-*/* {trainer.args.output_dir}"
    )


if __name__ == "__main__":
    main()
