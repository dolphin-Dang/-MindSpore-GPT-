"""
transform dataset to mindrecord.
"""

import argparse
import os
import pickle

import numpy as np
from mindspore.mindrecord import FileWriter
from tqdm.auto import tqdm

SEQ_LEN = 128

class Tokenizer:
    def __init__(self, data: str, block_size: int, tokenizer_path=None):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f"the fiction data has {data_size} characters, {vocab_size} unique.")
        
        if tokenizer_path != None:
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            self.stoi = tokenizer.stoi
            self.itos = tokenizer.itos
            cnt = 0
            for ch in chars:
                if ch not in self.stoi:
                    next_index = len(self.stoi)
                    self.stoi[ch] = next_index
                    self.itos[next_index] = ch
                    cnt += 1
            print(f"Added {cnt} new characters into tokenizer.")
        else:
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            print(f"Creating a new tokenizer.")
        self.block_sizes = block_size
        self.data = data

    def get_item(self):
        dix = [self.stoi[s] for s in self.data]
        for chunk in chunks(dix, self.block_sizes + 1):
            sample = {}
            if len(chunk) == self.block_sizes + 1:
                sample["input_ids"] = np.array(chunk, dtype=np.int32)
                yield sample


def chunks(lst, n):
    """yield n sized chunks from list"""
    for i in tqdm(range(len(lst) - n + 1)):
        yield lst[i : i + n]


def read_jinyong(file_path):
    """read Jin Yong fictions"""
    data = ""
    with open(os.path.join(file_path), "r", encoding="utf-8") as f:
        data += f.read().strip()

    return data


if __name__ == "__main__":
    local_config = {
        "fiction_name": "鸳鸯刀",
        "output_file": "./dataset/yyd.mindrecord",
        "tokenizer_path": "./dataset/tokenizer.pkl"
    }
    

    out_dir, out_file = os.path.split(os.path.abspath(local_config["output_file"]))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mindrecord_schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
    }

    data = read_jinyong("./data/"+ local_config["fiction_name"] +".txt")
    tokenizer = Tokenizer(data, block_size=SEQ_LEN, tokenizer_path=local_config["tokenizer_path"])
    with open("./dataset/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    transforms_count = 0
    wiki_writer = FileWriter(file_name=local_config["output_file"], shard_num=1)
    wiki_writer.add_schema(mindrecord_schema, "JinYong fictions")
    for x in tokenizer.get_item():
        transforms_count += 1
        wiki_writer.write_raw_data([x])
    wiki_writer.commit()
    print("Transformed {} records.".format(transforms_count))
    print("Transform finished, output files refer: {}".format(local_config["output_file"]))
