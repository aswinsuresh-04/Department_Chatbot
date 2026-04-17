import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for a university department. "
                    "Answer questions accurately and naturally using the information provided. "
                    "If the information is available, give a complete and helpful answer. "
                    "If the specific information asked is genuinely not available, say 'I don't have that information.' and stop — "
                    "do not substitute with unrelated information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content