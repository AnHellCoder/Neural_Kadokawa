import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


MODEL_PATH = "/userspace/kad2/models/qwen3-4b"
DATA_PATH = "/userspace/kad2/neural_kadokawa/prepared_dataset/prepared_messages.json"
OUTPUT_PATH = "/userspace/kad2/models/nk_lora"

MAX_LENGTH = 2048
SEED = 42


def main():
    model_path = Path(MODEL_PATH)
    data_path = Path(DATA_PATH)
    output_path = Path(OUTPUT_PATH)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Не найдена папка модели: {model_path}"
        )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Не найден датасет: {data_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. Запускай этот скрипт на GPU-ноде."
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
            "В датасете отсутствует колонка 'messages'. "
            "Проверь формат prepared_messages.jsonl."
        )

    print(f"Всего примеров: {len(dataset)}")
    print(f"Колонки: {dataset.column_names}")

    if len(dataset) < 2:
        raise ValueError(
            "Для разделения на train/eval нужно минимум 2 примера."
        )

    split_dataset = dataset.train_test_split(
        test_size=0.1,
        seed=SEED,
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

    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=str(output_path),

        num_train_epochs=15,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=0.05,
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
        seed=SEED,

        max_length=MAX_LENGTH,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,

        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],

        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Начинаем обучение...")
    trainer.train()

    output_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"Обучение завершено.")
    print(f"LoRA-адаптер сохранён в: {output_path}")


if __name__ == "__main__":
    main()
