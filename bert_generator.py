from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

#Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def fill_in_the_blanks(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    token_logits = model(input_ids).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_token = torch.topk(mask_token_logits, 1, dim=1).indices[0].item()
    input_ids[0, mask_token_index] = top_token
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "artificial intelligence is transforming [MASK] by"
    filled_text = fill_in_the_blanks(prompt)
    print(filled_text)    
