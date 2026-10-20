import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer
from transformers.trainer_utils import get_last_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-fune-tuning of Qwen", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("/userspace/kad2/models/qwen3-4b"), help="Base model path")
    parser.add_argument("--dataset", type=Path, default=Path("prepared_dataset/prepared_messages.json"), help="Dataset JSON")
    parser.add_argument("--savefolder", type=Path, default=Path("/userspace/kad2/models/nk_lora"), help="Output folder")
    parser.add_argument("--epochs", type=float, default=15, help="Target train epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--weights", type=Path, help="Fine-tuned LoRA-adapter for fine-tuning continuation")
    mode.add_argument("--resume", nargs="?", const="auto", help="Continue from checkpoint")
    args = parser.parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("epochs, learning-rate must be positive")
    if args.weights and not (args.weights / "adapter_config.json").is_file():
        parser.error("adapter_config.json not found")
    return args


def main():
    args = parse_args()
    model_path = args.model
    data_path = args.dataset
    output_path = args.savefolder
    checkpoint = None
    if args.resume:
        checkpoint = (get_last_checkpoint(str(output_path)) if output_path.is_dir() else None) if args.resume == "auto" else args.resume
        if checkpoint is None or not (Path(checkpoint) / "trainer_state.json").is_file():
            raise FileNotFoundError("Train checkpoint from --resume not found")

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script on GPU-node."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )

    if "messages" not in dataset.column_names:
        raise ValueError(
            "Dataset doesn't contain 'messages' column. "
            "Check prepared_messages.json."
        )

    print(f"Total examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    if len(dataset) < 2:
        raise ValueError(
            "From train/eval split at least  2 examples required."
        )

    split_dataset = dataset.train_test_split(
        test_size=0.1,
        seed=args.seed,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": 0},
    )

    if args.weights:
        model = PeftModel.from_pretrained(model, str(args.weights), is_trainable=True, local_files_only=True)

    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=str(output_path),

        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=0,
        lr_scheduler_type="cosine",

        logging_strategy="steps",
        logging_steps=10,

        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        bf16=use_bf16,
        fp16=not use_bf16,

        gradient_checkpointing=True,
        optim="adamw_torch",

        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,

        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,

        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],

        processing_class=tokenizer,
        peft_config=None if args.weights else lora_config,
    )

    print("Train started...")
    if checkpoint:
        print(f"Resume from: {checkpoint}")
    trainer.train(resume_from_checkpoint=checkpoint)

    output_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"Training is over.")
    print(f"LoRA-adapter saved into: {output_path}")


if __name__ == "__main__":
    main()
