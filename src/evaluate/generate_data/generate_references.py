import json
import re
import os

INPUT_PATH = "dataset/test.jsonl"
OUTPUT_PATH = "data/references.jsonl"


def extract_text(messages):
    for m in messages:
        if m["role"] == "assistant":
            return m["content"][0]["text"]
    return ""


def parse_section(text, section):
    pattern = rf"{section}:\s*- Material:\s*(.*?)\s*- Kondisi:\s*(.*?)\n"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "tidak_terlihat", "tidak_terlihat"


def parse_konflik(text):
    match = re.search(r"Konflik:\s*- Dinding:\s*(true|false)", text)
    if match:
        return match.group(1) == "true"
    return False


def parse_penjelasan(text):
    match = re.search(r"Penjelasan:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def main():
    os.makedirs("data", exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            text = extract_text(data["messages"])

            atap_m, atap_k = parse_section(text, "Atap")
            dinding_m, dinding_k = parse_section(text, "Dinding")
            lantai_m, lantai_k = parse_section(text, "Lantai")

            konflik = parse_konflik(text)
            penjelasan = parse_penjelasan(text)

            # 🔥 FORMAT OUTPUT SESUAI PARSER
            output_text = f"""Atap:
- Material: {atap_m}
- Kondisi: {atap_k}

Dinding:
- Material: {dinding_m}
- Kondisi: {dinding_k}

Lantai:
- Material: {lantai_m}
- Kondisi: {lantai_k}

Konflik Dinding: {"Ya" if konflik else "Tidak"}

Penjelasan:
{penjelasan}
"""

            result = {
                "id": data["id"],
                "output": output_text.strip()
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("✅ references.jsonl siap untuk evaluasi!")


if __name__ == "__main__":
    main()