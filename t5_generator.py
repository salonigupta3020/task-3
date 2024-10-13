from transformers import T5Tokenizer, T5ForConditionalGeneration

#Load pre-trained T5 model and tokenizer
model_name = 't5-small'
model = T5ForConditionalGeneration.form_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_article(prompt):
    input_text = f"generate article: {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Artificial intelligence is transforming industried by"
    article = generate_article(prompt)
    print(article)
