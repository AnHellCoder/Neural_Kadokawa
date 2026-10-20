import argparse


def main():
    parser = argparse.ArgumentParser(description="Anime titles generation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="/userspace/kad2/models/qwen3-4b", help="Base model path")
    parser.add_argument("--weights", default="/userspace/kad2/models/nk_lora", help="LoRA-adapter")
    parser.add_argument("--prompt", required=True, help="Request")
    parser.add_argument("--count", type=int, default=50, help="Amount of generated titles")
    parser.add_argument("--temperature", type=float, default=0.98, help="Temperature")
    args = parser.parse_args()
    if args.count < 1 or args.temperature <= 0:
        parser.error("count and temperature must be positive")
    if not args.prompt.strip():
        parser.error("Prompt is empty")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.weights, local_files_only=True)
    model.eval()
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
    input_length = inputs.input_ids.shape[1]
    for index in range(1, args.count + 1):
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=528, do_sample=True, temperature=args.temperature, top_p=0.95, top_k=60, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
        content = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
        print(f"===== Generation {index} =====", flush=True)
        print(content, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
