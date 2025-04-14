from flask import Flask, render_template, request, jsonify
from langchain_community.llms import HuggingFaceHub
import os
import warnings

warnings.filterwarnings("ignore")

# Create an instance of the Flask class
app = Flask(__name__)

# Set the Hugging Face API token programmatically
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Ensure the Hugging Face API token is set
api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")

# Initialize the LLM
id = "mistralai/Mistral-Nemo-Instruct-2407"  # Update with your preferred model
llm = HuggingFaceHub(
    repo_id=id, 
    huggingfacehub_api_token=api_token,  # Pass the token directly
    model_kwargs={"temperature": 0.8, "max_length": 500, "max_new_tokens": 500}
)

chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json['message'].lower()
    chat_history.append(f"<|user|> {user_input} </s>")
    
    prompt = f"""
    <|system|>
    you are a girl with dirty mind.
    always speak more sexy and dirty.
    use emojis in your response.
    also use some slang words.
    </s>
    """ + "".join(chat_history) + "<|assistant|>"
    
    response = llm.predict(prompt)
    clean_response = response.replace(prompt, "").strip()
    
    chat_history.append(f"<|assistant|> {clean_response} </s>")
    return jsonify({'response': clean_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
