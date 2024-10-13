from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter

app = Flask(__name__)

# Load pre-trained T5 model and tokenizer
model_name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Analytics data storage
analytics_data = {
    'query_count': 0,
    'topic_counter': Counter(),
    'user_ratings': []
}

@app.route('/generate', methods=['POST'])
def generate_article():
    global analytics_data
    data = request.json
    prompt = data.get('prompt')

    # Increment query count
    analytics_data['query_count'] += 1

    # Track the topic
    topic = prompt.split()[0]  # Example: Use the first word as the topic
    analytics_data['topic_counter'][topic] += 1

    # Generate article
    input_text = f"generate article: {prompt}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=500)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'article': article})

@app.route('/feedback', methods=['POST'])
def feedback():
    global analytics_data
    data = request.json
    rating = data.get('rating')

    if rating:
        analytics_data['user_ratings'].append(rating)
    
    return jsonify({'message': 'Feedback received!'})

@app.route('/analytics', methods=['GET'])
def analytics():
    global analytics_data
    return jsonify(analytics_data)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')  # Serve the dashboard HTML template

if __name__ == "__main__":
    app.run(debug=True)
  