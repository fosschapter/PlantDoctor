# chat_app.py
from groq import Groq

client = Groq(api_key="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")

def chat_with_bot(message, history):
    if not message.strip():
        return history + [["", "Please ask a question about plant diseases or treatments."]]

    try:
        messages = [
            {"role": "system", "content": "You are an expert in plant diseases and treatments. Answer questions based on agricultural knowledge."},
            {"role": "user", "content": message}
        ]

        completion = client.chat.completions.create(
            model="qwen-2.5-32b",
            messages=messages,
            temperature=0.5,
            max_completion_tokens=250,
            top_p=1.0,
            stream=True,
        )

        response = ""
        for chunk in completion:
            if "choices" in chunk and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        if not response.strip():
            response = "I couldn't generate an answer. Please try rephrasing your question or ask a different one."

        history.append([message, response])
    except Exception as e:
        history.append([message, f"Error communicating with Groq API: {e}"])

    return history
