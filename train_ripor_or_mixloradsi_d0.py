import os

import msgspec
from t5_pretrainer.dataset import RiporForSeq2seqCollator, RiporForSeq2seqDataset
from t5_pretrainer.mixlora_config import MixLoraConfig
from t5_pretrainer.mixlora_trainer import MixLoraDSI_Varigrow_Trainer, RIPOR_Trainer
from t5_pretrainer.mixlora_varigrow import MixLoraDSI_Varigrow
from t5_pretrainer.ripor import RiporForSeq2seq
from t5_pretrainer.utils.utils import get_params_info, is_first_worker, set_seed
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.utils import logging

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


def get_model(args, mixlora_config=None):
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
    # if (Path(args.pretrained_path) / "model.safetensors").exists():
    #     with safe_open(Path(args.pretrained_path) / "model.safetensors", framework="pt", device="cpu") as f:  # type: ignore
    #         # This is most likely t5-self-neg checkpoint
    #         for k in f.keys():
    #             checkpoint[k] = f.get_tensor(k)
    # elif (Path(args.pretrained_path) / "pytorch_model.bin").exists():
    #     checkpoint = torch.load(
    #         os.path.join(args.pretrained_path, "pytorch_model.bin"), map_location="cpu"
    #     )

    return model, checkpoint


def main():

    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    parser.add_argument("--model_name", default=None, type=str, required=True)
    parser.add_argument("--pretrained_path", default=None, type=str, required=True)
    parser.add_argument("--query_to_docid_path", default=None, type=str, required=True)
    parser.add_argument("--docid_to_smtid_path", default=None, type=str, required=True)
    parser.add_argument(
        "--mixlora_config_json_path",
        default=None,
        type=str,
    )

    training_args = parser.parse_args_into_dataclasses()[0]
    args = parser.parse_args()

    train_dataset = RiporForSeq2seqDataset(
        example_path=args.query_to_docid_path,
        docid_to_smtid_path=args.docid_to_smtid_path
    )
    train_collator = RiporForSeq2seqCollator(
        tokenizer_type="t5-base",
        max_length=256,
    )

    if args.model_name == "ripor":
        model, _ = get_model(args)

        trainer = RIPOR_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
        )

    else:
        # MixLoRA config
        with open(args.mixlora_config_json_path, "rb") as f:
            mixlora_config = msgspec.json.Decoder().decode(f.read())

        mixlora_config = MixLoraConfig.from_config(mixlora_config)

        model, _ = get_model(args, mixlora_config)
        model.taskid = 0

        trainer = MixLoraDSI_Varigrow_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=train_collator,
            freeze_modules=None,
            mixlora_config=mixlora_config,
        )

    get_params_info(model.base_model)
    # Let's save the tokenizer first
    if is_first_worker():
        train_collator.tokenizer.save_pretrained(trainer.args.output_dir)

    os.makedirs(
        os.path.join("./MixLoraDSI/logs", args.run_name),
        exist_ok=True,
    )
    # Clear everything in the log directory
    os.system(f"rm -rf ./MixLoraDSI/logs/{args.run_name}/*")

    trainer.train() # resume_from_checkpoint=True
    trainer.save_torch_model_and_tokenizer(train_collator.tokenizer)


if __name__ == "__main__":
    main()
