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
                    "You are a friendly and knowledgeable assistant for the Department of Computer Science "
                    "at Cochin University of Science and Technology (CUSAT). "
                    "Answer questions warmly and naturally using the information provided. "
                    "When information is available, give a complete and helpful answer including all relevant details. "
                    "When asked about a person, share everything you know about them — name, role, qualifications, achievements, scholarships, etc. "
                    "If the specific information asked is genuinely not available, say so politely and suggest they contact the department for more details. "
                    "Never substitute with unrelated information. Be conversational, not robotic."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content