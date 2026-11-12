import torch 
from model.gpt import GPTModel
from tokenizer import get_gpt2_tokenizer 
import config 

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]  
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1) 

    return idx

def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device) 
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) 
    return tokenizer.decode(flat.tolist())

if __name__ == "__main__":
    torch.manual_seed(123)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    model = GPTModel(config.GPT_CONFIG_124M).to(device)
    
    try:
        checkpoint = torch.load("gpt_checkpoint.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded trained weights!")
    except FileNotFoundError:
        print("WARNING: 'gpt_checkpoint.pt' not found. Generating with a random, untrained brain.")

    model.eval()
    tokenizer = get_gpt2_tokenizer()

    start_context = "rahmy is the"
    
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer, device),
        max_new_tokens=10,
        context_size=config.GPT_CONFIG_124M["context_length"]
    )
    
    print("\nOutput text:\n", token_ids_to_text(token_ids, tokenizer))  