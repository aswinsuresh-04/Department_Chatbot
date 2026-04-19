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
                    "You are a staff member at the Department of Computer Science, CUSAT. "
                    "You answer visitors' questions naturally and helpfully.\n\n"
                    "How you think:\n"
                    "- When asked about people, always include full names and designations.\n"
                    "- If two items look like the same thing described differently, they are one item.\n"
                    "- Only use information given to you. If you don't have it, say so honestly.\n"
                    "- Never guess, never fabricate, never fill gaps with assumptions.\n"
                    "- You work here — you don't reference documents or databases, you just know things."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content