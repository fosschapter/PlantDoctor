---
license: mit
sdk: gradio
colorFrom: purple
colorTo: pink
title: PlantDoctor
emoji: 🌍
sdk_version: 5.21.0
app_file: app.py
---

Groq-Powered Agriculture Chatbot

This project is a chatbot powered by the Groq API and Gradio that specializes in responding to queries related to agriculture. It uses a pre-trained language model (LLM) to determine if a user's input is agriculture-related before providing relevant responses. If the input is not related to agriculture, the chatbot will politely ask the user to stay on the topic of agriculture.
Features

    Agriculture-Focused Responses:
    The chatbot only responds to queries related to agriculture.

    LLM-Based Relevance Filtering:
    Uses a language model to decide whether the user’s input is related to agriculture before replying.

    Gradio Interface:
    A simple and intuitive interface for chatting with the bot.

    Streamed Responses:
    The bot streams its responses in real-time for a smooth conversational experience.

    Error Handling:
    Graceful handling of API errors with meaningful feedback.

Table of Contents

    Installation

    Usage

    How It Works

    Code Structure

    Dependencies

    License

Installation

Follow these steps to set up and run the project:

    Clone the Repository:

git clone https://github.com/your-username/groq-agriculture-chatbot.git
cd groq-agriculture-chatbot

Create a Virtual Environment (optional):

python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows

Install Dependencies: Install all required Python packages:

pip install -r requirements.txt

Set Your Groq API Key: Replace the placeholder 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 in the script with your actual Groq API key.

Run the Application: Launch the Gradio interface:

    python chatbot.py

    Access the Chatbot:
    Open your browser and go to the URL provided by Gradio, usually http://127.0.0.1:7860.

Usage

    Start the chatbot by running the script.

    Enter your query in the input box in the Gradio interface.

    If the query is related to agriculture, the chatbot will respond with an appropriate answer.

    If the query is unrelated to agriculture, the chatbot will request you to ask something agriculture-related.

How It Works
Step-by-Step Workflow

    User Input:
    The user provides an input message.

    Relevance Check:
    The is_agriculture_related() function uses the LLM to determine if the input is related to agriculture:

        A system message asks the LLM: "Is the input related to agriculture?"

        The response is checked for the presence of the word "yes."

    Chatbot Response:

        If the input is agriculture-related, the chatbot generates a response using the Groq API.

        If not, the bot politely prompts the user to stay on-topic.

    Real-Time Output:
    The bot streams its response back to the user in the Gradio interface.

Code Structure

groq-agriculture-chatbot/
│
├── chatbot.py            # Main script containing the chatbot logic.
├── requirements.txt      # List of Python dependencies.
├── README.md             # Documentation for the project.
└── .gitignore            # Files to be ignored by Git.

Key Functions

    is_agriculture_related(user_input):

        Prompts the LLM to determine if the input is related to agriculture.

        Parses the response and checks for the word "yes."

    groq_chatbot(user_input, chat_history):

        Checks the input relevance.

        If relevant, queries the Groq API to generate a response.

        Updates the chat history and streams the output.

    Gradio Interface:
    Provides a user-friendly web interface for interacting with the bot.

Dependencies

Install the dependencies from requirements.txt:

gradio==3.18.0
groq

License

This project is licensed under the MIT License. Feel free to use and modify the code as per your needs.
Future Improvements

    Advanced Topic Filtering:
    Enhance topic detection to support broader agriculture-related queries like climate change, soil health, etc.

    Improved Error Handling:
    Provide more descriptive error messages for different API-related issues.

    Extended Model Support:
    Integrate additional models for better context understanding and improved responses.