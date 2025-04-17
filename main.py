from langchain.llms import HuggingFaceHub
import os
import warnings
warnings.filterwarnings("ignore")
from getpass import getpass
os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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" 


chat_history=[]

id="mistralai/Mistral-Nemo-Instruct-2407"
#id="HuggingFaceH4/zephyr-7b-beta"
#id="qihoo360/TinyR1-32B-Preview"
llm=HuggingFaceHub(repo_id=id, model_kwargs={"temperature":0.8, "max_length":50000, "max_new_tokens":50000})

query = ""
while query != "exit":
    query = input("You:").lower()
    chat_history.append(f"<|user|> {query} </s>")
    prompt = f"""
    <|system|>
    You are a girl with dirty mind,
    So talk accordingly.
    </s>
    """ + "".join(chat_history)+"" + "<|assistant|>"

    response = llm.predict(prompt)
    clean_response = response.replace(prompt, "").strip()
    print("ChatBot:", clean_response)
    chat_history.append(f"<|assistant|> {clean_response} </s>")
