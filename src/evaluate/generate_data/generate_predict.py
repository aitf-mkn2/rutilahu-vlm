import json

input_path = "../data/predictions_inference.jsonl"
output_path = "../data/predictions.jsonl"

def fix_prediction(text):
    text = text.replace("\r", "")

    # Fix konflik format
    if "Konflik:" in text:
        if "false" in text.lower():
            text = text.replace("Konflik:\n- Dinding: false", "Konflik Dinding: false")
        elif "true" in text.lower():
            text = text.replace("Konflik:\n- Dinding: true", "Konflik Dinding: true")

    return text

new_lines = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        new_data = {
            "id": data["id"],
            "output": fix_prediction(data["prediction"])
        }

        new_lines.append(json.dumps(new_data, ensure_ascii=False))

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

print("DONE")