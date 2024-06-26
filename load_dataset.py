import json
import random
import sys

from datasets import load_dataset
from tqdm import tqdm
from bad_substrings import has_bad_ss

def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])

def main(train_path, val_path):
    random.seed(42)

    instruct_records = []
    for row in tqdm(load_dataset("lksy/ru_instruct_gpt4", split="train")):
        message = row["instruction"]
        if row["input"]:
            message += "\nДано: " + row["input"]
        output = row["full_output"]
        if not output:
            continue
        if has_bad_ss([{"content": output}]):
            continue
        instruct_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": "gpt4"
        })
    print("Instruct gpt4 count:", len(instruct_records))
    print("Instruct gpt4 length:", calc_max_length(instruct_records))

    saiga_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_saiga", split="train")):
        messages = revert_flattening(row["messages"])
        if has_bad_ss(messages):
            continue
        saiga_records.append({
            "messages": messages,
            "source": "saiga"
        })
    print("Saiga count:", len(saiga_records))
    print("Max Saiga length:", calc_max_length(saiga_records))

    merged_instruct_records = []
    prev_record_idx = None
    for idx, record in enumerate(instruct_records):
        text_length = sum([len(m["content"]) for m in record["messages"]])
        if text_length > 1000:
            merged_instruct_records.append(record)
            continue
        if prev_record_idx is None:
            prev_record_idx = idx
            continue
        messages = instruct_records[prev_record_idx]["messages"] + record["messages"]
        merged_instruct_records.append({
            "messages": messages,
            "source": "merged_instruct"
        })
        prev_record_idx = None
    print("Merged instruct count:", len(merged_instruct_records))
    print("Max Merged instruct length:", calc_max_length(merged_instruct_records))
    instruct_records = merged_instruct_records

    records = saiga_records
    excluded_indices = set()
    for record in tqdm(instruct_records):
        text_length = sum([len(m["content"]) for m in record["messages"]])
        if text_length > 2000 or random.random() < 0.5:
            records.append(record)
            continue
        index = random.randrange(len(records))
        while index in excluded_indices:
            index = random.randrange(len(records))
        excluded_indices.add(index)
        records[index]["source"] = "mixed"
        records[index]["messages"] += record["messages"]
    print("Saiga + instruct count:", len(records))
    print("Max Saiga + instruct length:", calc_max_length(records))

    random.shuffle(records)
    border = int(0.95 * len(records))
    train_records = records[:border]
    val_records = records[border:]
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + ",\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + ",\n")

train_path = sys.argv[1]
val_path = sys.argv[2]
print(sys.argv)
main(train_path, val_path)