from transformers import GPT2LMHeadModel, GPT2Tokenizer

#Load pre-trained GPT-2 model and tokenizer
model_name ='gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_artile(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Artificial intelligence is transforming industry by"
    article = generate_artile(prompt)
    print(article)