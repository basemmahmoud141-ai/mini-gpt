import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm
import numpy as np


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt, allowed_special="all")

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        self.input_ids = []
        self.target_ids = []

        num_chunks = max(0, (len(token_ids) - max_length) // stride)

        for i in tqdm(range(0, len(token_ids) - max_length, stride), desc="Chunking", unit="chunk"):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def get_gpt2_tokenizer():
    return tiktoken.get_encoding("gpt2")


def create_text_dataloader_v1(txt, batch_size=16, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = get_gpt2_tokenizer()
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    use_pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    return dataloader


class MemmapDataset(Dataset):
    def __init__(self, bin_path, max_length, stride, dtype=np.uint16):
        self.data = np.memmap(bin_path, dtype=dtype, mode='r')
        self.max_length = max_length
        self.stride = stride
        self.num_chunks = max(0, (len(self.data) - self.max_length) // self.stride)

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        chunk = self.data[start_idx : start_idx + self.max_length + 1]
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))

        input_ids = chunk_tensor[:-1]
        target_ids = chunk_tensor[1:]

        return input_ids, target_ids


def create_dataloader_v1(bin_path, batch_size=16, max_length=256, stride=128, shuffle=True, drop_last=True):
    dataset = MemmapDataset(bin_path, max_length, stride)
    use_pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    return dataloader


def create_jsonl_dataloaders(jsonl_path ,batch_size=16,max_length=256,stride=128,val_ratio=0.1,text_key="text",shuffle_train=True,drop_last_train=True,):    
    tokenizer = get_gpt2_tokenizer()
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj[text_key])


    split_idx = int(len(texts) * (1 - val_ratio))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    train_text = "<|endoftext|>".join(train_texts)
    val_text = "<|endoftext|>".join(val_texts)


    train_dataset = GPTDatasetV1(train_text, tokenizer, max_length, stride)
    val_dataset = GPTDatasetV1(val_text, tokenizer, max_length, stride)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last_train,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader