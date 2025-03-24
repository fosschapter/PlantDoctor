import gradio as gr
from groq import Groq

class GroqChatbot:
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def respond(self, user_input, chat_history=[]):
        try:
            # Prepare messages for the Groq API
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for user_message, bot_message in chat_history:
                messages.append({"role": "user", "content": user_message})
                messages.append({"role": "assistant", "content": bot_message})
            messages.append({"role": "user", "content": user_input})

            # Call the Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1,
                max_completion_tokens=250,
                top_p=1,
                stream=True,
                stop=None,
            )

            # Collect the response from the streaming output
            bot_reply = ""
            for chunk in completion:
                bot_reply += chunk.choices[0].delta.content or ""

            # Update the chat history
            chat_history.append((user_input, bot_reply))
            return "", chat_history
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Error: {str(e)}"
            chat_history.append((user_input, error_message))
            return "", chat_history

    def create_gradio_ui(self):
        with gr.Blocks() as gradio_app:
            gr.Markdown("# Chatbot Using Groq API")
            chat_history_state = gr.State([])

            with gr.Row():
                chatbot = gr.Chatbot()
                message_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your Message"
                )
                send_button = gr.Button("Send")

            send_button.click(
                fn=self.respond,
                inputs=[message_input, chat_history_state],
                outputs=[message_input, chatbot],
                show_progress=True,
            )

        return gradio_app
