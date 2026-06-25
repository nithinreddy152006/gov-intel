import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print(f"Checking with key: {api_key[:5]}...{api_key[-5:]}")
try:
    for m in genai.list_models():
        print(m.name)
except Exception as e:
    print(f"Error: {e}")
