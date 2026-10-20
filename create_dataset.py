import json
import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "prepared_dataset"

RUSSIAN_INPUT_PATH = BASE_DIR / "data" / "title_russian.txt"
RUSSIAN_OUTPUT_PATH = OUTPUT_DIR / "prepared_messages.json"

JAPANESE_INPUT_PATH = BASE_DIR / "data" / "anime_cleaned.csv"
JAPANESE_OUTPUT_PATH = (
    OUTPUT_DIR / "prepared_messages_japanese.json"
)

RUSSIAN_USER_PROMPT = (
    "Напиши название аниме-тайтла на русском языке"
)

JAPANESE_USER_PROMPTS = [
    "Напиши название аниме-тайтла на японском языке",
    "日本語でアニメのタイトルを書いてください",
]


def create_message(prompt, title):
    return {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": title,
            },
        ]
    }


def save_dataset(messages, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            messages,
            file,
            ensure_ascii=False,
            indent=2,
        )


# Русские тайтлы.

prepared_messages = []

with RUSSIAN_INPUT_PATH.open(
    "r",
    encoding="utf-8-sig",
) as file:
    for line in file:
        title = line.strip()

        if title:
            prepared_messages.append(
                create_message(RUSSIAN_USER_PROMPT, title)
            )


# Японские тайтлы.

df = pd.read_csv(
    JAPANESE_INPUT_PATH,
    usecols=["title_japanese"],
    dtype={"title_japanese": "string"},
    encoding="utf-8-sig",
)

japanese_titles = df["title_japanese"].dropna().str.strip()
japanese_titles = (
    japanese_titles[japanese_titles.ne("")]
    .drop_duplicates()
    .tolist()
)

# Перемешивание воспроизводимо благодаря фиксированному seed.
rng = random.Random(42)
rng.shuffle(japanese_titles)

prepared_messages_japanese = [
    create_message(
        JAPANESE_USER_PROMPTS[
            index % len(JAPANESE_USER_PROMPTS)
        ],
        title,
    )
    for index, title in enumerate(japanese_titles[:len(prepared_messages)])
]


# Сохранение обоих датасетов.

save_dataset(prepared_messages+prepared_messages_japanese, RUSSIAN_OUTPUT_PATH)
#save_dataset(
#    prepared_messages_japanese,
#    JAPANESE_OUTPUT_PATH,
#)


# Проверка результата.

print("Русский датасет:")
print(f"Подготовлено записей: {len(prepared_messages)}")
print(f"Результат сохранён в: {RUSSIAN_OUTPUT_PATH}")

print("\nЯпонский датасет:")
print(
    f"Подготовлено записей: "
    f"{len(prepared_messages_japanese)}"
)
print(f"Результат сохранён в: {JAPANESE_OUTPUT_PATH}")

for prompt in JAPANESE_USER_PROMPTS:
    count = sum(
        message["messages"][0]["content"] == prompt
        for message in prepared_messages_japanese
    )
    print(f"Запрос {prompt!r}: {count} записей")

if prepared_messages:
    ratio = (
        len(prepared_messages_japanese)
        / len(prepared_messages)
    )
    print(
        f"\nСоотношение японских и русских записей: "
        f"{ratio:.2f}:1"
    )

if prepared_messages_japanese:
    print("\nПервые две японские записи:")
    print(json.dumps(
        prepared_messages_japanese[:2],
        ensure_ascii=False,
        indent=2,
    ))
