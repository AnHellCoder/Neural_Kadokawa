# Neural Kadokawa
## Abstract:
Neural Kadokawa is an independent project inspired by Neural Meduza. It uses a LoRA-fine-tuned Qwen3-4B model to generate absurd anime-style titles in Russian and Japanese.
## Fine-tuning and dataset
The model is fine-tuned on 1569 training examples for 15 epochs. The dataset contains three types of prompts:
1. Prompt in Russian for Russian names generation (523 messages in Russian with 523 Russian-language assistant responses)
2. Prompt in Japanese for Japanese names generation (523 messages in Japanese with 523 Japanese-language assistant responses)
3. Prompt in Russian for Japanese names generation (523 messages in Russian with 523 Japanese-language assistant responses)

Dataset sources:
1. MyAnimeList Dataset: https://www.kaggle.com/datasets/azathoth42/myanimelist (for Japanese names)
2. Dream Cast telegram channel: https://t.me/dreamercast (for Russian names)
Generation model: https://huggingface.co/Qwen/Qwen3-4B
## Generation examples:
Generated titles are not checked for uniqueness and may overlap with existing works or reference existing franchises.
### Prompt: Напиши название аниме-тайтла на русском языке. Напиши только название, больше ничего.
Приключения и квест «Головоломка идёт, голова моя бьётся»
Япония, сбежавшая из 21 века
Доходная магическая команда 2.0
Незнакомец, который стал тёмным колоритным телёнком в аниме-тайтле для манги под названием «Начинай рассказывать мне сюжет!» (типа)
### Prompt: 日本語でアニメのタイトルだけを書いてください 
アタマが割れたらどうする? ～ドクターリボン物語～
学園ヒロイン ジャイアント・ストライク! 〜ドッコイドッコドッコイ〜
アイアンヒーロー（日本語吹き替え版）
乙女狩り! 〜ポケモンカード戦記〜
このように、私は生きている。
### Prompt: Напиши название аниме-тайтла на японском языке. Используй ТОЛЬКО японский язык.
アルパカ大戦争!! マヨマヨはロクでなし
おだやかにドメスティックな日常 ♡ その他のキャラも可愛いんでちょっとしか見れません おだやかにドメスティックな日常 ♡
かめはじか! 俺を倒してしまえ
恋伽 サクラが爆走する ～魔道書から逃げる男～

## Results:
The model generates anime-style titles in Russian and Japanese quite well, some outputs are amusing and coherent, though some titles may be awkward, repetitive, nonsensical, or closely resemble existing titles. Future work will focus on improving coherence and humor while reducing repetition and memorization.

**Contributions are welcome: report bugs, suggest dataset improvements, or submit amusing generated titles through an issue or pull request.**
## How to run this project:
1. The raw data is presented in data folder with two text files: "titles_russian.txt" and "titles_japanese.txt". For now, they are the only sources of dataset. You may edit them if you would like to train your own model.
2. Running "create_dataset.py", you may process raw data to prepared dataset. By default, the dataset is saved in **prepared_dataset/prepared_messages.json** but you can write it into other file (provide filename to "--savefile" argument). **Warning:** The training script loads datasets only from .json files so you must write your prepared dataset into .json file.
3. Run **train_qwen_lora.py** from **train** folder. By default, train script reads dataset from **prepared_messages.json** file, trains for 15 epochs and saves weights to **nk_lora** folder. You can load dataset from other file by providing "--dataset" argument, set your number of epochs by providing "--epochs" argument and specify the output directory by providing "--savefolder" argument. Also, you can continue fine-tuning the model from a checkpoint by providing "--weights" argument. The trained LoRA adapter is included in the repository.
4. Run **generate_name.py** to generate names and provide "--prompt" argument to set a prompt to the model (using prompts from **Generation examples** is recommended) and "--output-length" argument to define particular number of generated name. If you find some generated names funny, you may send them to a.kuropatov1@g.nsu.ru, and I will add them to the raw data for future fine-tuning.
