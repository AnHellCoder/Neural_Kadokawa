import argparse
import json
import random
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preparation of common RU/JA dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--russian-input", type=Path, default=root / "data/title_russian.txt", help="Russian titles, TXT")
    parser.add_argument("--japanese-input", type=Path, default=root / "data/title_japanese.txt", help="Japanese titles, TXT")
    parser.add_argument("--savefile", type=Path, default=root / "prepared_dataset/prepared_messages.json", help="Output JSON")
    args = parser.parse_args()
    if args.savefile.suffix.lower() != ".json":
        parser.error("--savefile must have .json")
    import pandas as pd

    def message(prompt, title):
        return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": title}]}

    with args.russian_input.open(encoding="utf-8-sig") as file:
        russian = [line.strip() for line in file if line.strip()]
    with args.japanese_input.open(encoding="utf-8-sig") as file:
        japanese = [line.strip() for line in file if line.strip()]
    if not russian:
        parser.error("Russian txt file is empty")
    if not japanese:
        parser.error("Japanese txt file is empty")
    prompts = ["Напиши название аниме-тайтла на японском языке", "日本語でアニメのタイトルを書いてください"]
    records = [message("Напиши название аниме-тайтла на русском языке", title) for title in russian]
    records += [message(prompts[i % 2], title) for i, title in enumerate(japanese)]
    args.savefile.parent.mkdir(parents=True, exist_ok=True)
    with args.savefile.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    print(f"Russian-language responses: {len(russian)}; Japanese-language: {len(japanese)}; total: {len(records)}")
    print(f"Japanese-language responses: Russian-language request — {(len(japanese) + 1) // 2}, Japanese-language request — {len(japanese) // 2}")
    print(f"Saved: {args.savefile.resolve()}")


if __name__ == "__main__":
    main()
