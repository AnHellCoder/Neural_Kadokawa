import torch
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_PATH = "/userspace/kad2/models/qwen3-4b"
LORA_ADAPTER_PATH = "/userspace/kad2/models/nk_lora"


# Загружаем токенайзер базовой модели
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

# Загружаем базовую модель
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    dtype="auto",
    device_map="auto",
)

# Подключаем обученный LoRA-адаптер
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH,
)

model.eval()


# Подготавливаем вход
#prompt = "Напиши энциклопедическую статью о вымышленном существе."
#prompt = """
#Напиши сатирическую псевдоэнциклопедическую статью
#о полностью несуществующем божестве.

#Важно:
#- божество должно быть вымышленным;
#- не используй реальных людей, богов, царей и исторические события;
#- придумай нелепую область покровительства;
#- добавь абсурдный культ и бессмысленный ритуал;
#- текст должен звучать уверенно и академично, но содержание должно быть смешным;
#- не пиши обычную биографию врача, учёного, правителя или писателя;
#- не добавляй реальные ISBN, настоящие университеты и реальные библиографические источники.
#"""
def get_prompt():
     parser = argparse.ArgumentParser()

     parser.add_argument("--prompt")

     args = parser.parse_args()

     return args.prompt

prompt = get_prompt()
print(f"Подготовленный промт: {prompt}")

messages = [
    {
        "role": "user",
        "content": prompt,
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt",
).to(model.device)

for i in range(50):
    # Генерация
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=528,

            do_sample=True,
            temperature=0.98,
            top_p=0.95,
            top_k=60,
            repetition_penalty=1.05,

            pad_token_id=tokenizer.eos_token_id,
        )


    # Убираем токены исходного prompt
    input_length = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_length:].tolist()


    # Для enable_thinking=False отдельное извлечение thinking обычно не нужно
    content = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip()

    print(f"=====Генерация {i}=====")
    print(content)
    print()
