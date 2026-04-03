import gradio as gr
from openai import OpenAI

client = OpenAI()


def chat(message, history):
    """Stream responses from the OpenAI ChatCompletion API."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    partial = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            partial += delta
            yield partial


demo = gr.ChatInterface(
    fn=chat,
    title="ChatGPT Chatbot",
    description="A ChatGPT-like chatbot powered by OpenAI API and Gradio.",
    theme="soft",
    examples=["Hello!", "Explain quantum computing in simple terms", "Write a short poem about coding"],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
