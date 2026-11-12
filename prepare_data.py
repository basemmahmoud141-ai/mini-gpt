import os
import json
import numpy as np
import tiktoken
from tqdm import tqdm

def pre_tokenize_jsonl(input_path, output_path, text_field="text", chunk_size=100_000):
    """
    Reads a JSONL file, tokenizes the text, and writes to a binary file continuously.
    """
    print(f"Processing: {input_path}")
    print(f"Outputting to: {output_path}")
    
    # Initialize tokenizer (Assuming standard GPT-2 tokenizer. Change if using LLaMA/Custom)
    enc = tiktoken.get_encoding("gpt2")
    
    # End of text token to separate documents (GPT-2 standard is 50256)
    eot_token = enc.eot_token if hasattr(enc, 'eot_token') else 50256 

    # Count lines for the progress bar
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    
    # We use uint16 because GPT-2 vocab (50,257) fits inside 16-bit integers (max 65,535).
    # If using GPT-4 tokenizer (vocab 100k), change this to np.uint32!
    dtype = np.uint16
    
    tokens_buffer = []
    total_tokens = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'wb') as f_out:
        for line in tqdm(f_in, total=total_lines, desc="Tokenizing"):
            line = line.strip()
            if not line:
                continue
                
            obj = json.loads(line)
            text = obj[text_field]
            
            # Tokenize and append the End-Of-Text token
            tokens = enc.encode(text, allowed_special="all")
            tokens.append(eot_token)
            tokens_buffer.extend(tokens)
            
            # Write to disk in chunks to save RAM
            if len(tokens_buffer) >= chunk_size:
                np_array = np.array(tokens_buffer, dtype=dtype)
                f_out.write(np_array.tobytes())
                total_tokens += len(tokens_buffer)
                tokens_buffer = []

        # Write any remaining tokens
        if tokens_buffer:
            np_array = np.array(tokens_buffer, dtype=dtype)
            f_out.write(np_array.tobytes())
            total_tokens += len(tokens_buffer)
            
    print(f"Done! Total tokens written: {total_tokens:,}\n")

if __name__ == "__main__":
    # Define your paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "processed")
    
    train_in = os.path.join(data_dir, "train.jsonl")
    train_out = os.path.join(data_dir, "train.bin")
    
    val_in = os.path.join(data_dir, "val.jsonl")
    val_out = os.path.join(data_dir, "val.bin")
    
    # Run the processing
    if os.path.exists(train_in):
        pre_tokenize_jsonl(train_in, train_out)
    else:
        print(f"Could not find {train_in}")
        
    if os.path.exists(val_in):
        pre_tokenize_jsonl(val_in, val_out)