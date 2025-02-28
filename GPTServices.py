import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

try:
    openai_client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
    )
except Exception as e:
    print(f"[OPENAI] Error:({e}), please check whether you have the correct OPENAI_API_KEY setup within .env!")
    raise



def gpt_generate_single_response(user_prompt:str, system_prompt:str, model:str="gpt-4o", temperature=0.3, token_limit=500):
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=token_limit,
        )
        print(f"[OPENAI] Successfully generated gpt response: {response}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"[OPENAI] ERROR:Unable to generate GPT responses! ({e})")
        raise


def gpt_generate_embedding(text: str, model: str = "text-embedding-3-large"):
    response = openai_client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
    

def gpt_stream_responses(conversation, update_callback, finished_callback, chat_history):
    try:
        chat_history.add_user_message(conversation[-1]["content"])
        # Create a chat completion with streaming enabled.
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history.get_conversation(),
            stream=True
        )
        full_response = ""
        for chunk in response:
            # Each chunk is a dict. The text delta is found under: chunk.choices[0].delta.content
            delta = chunk.choices[0].delta
            content = delta.content

            if content:
                full_response += content
                # Pass each new chunk to the UI update callback.
                update_callback(content)
            # When complete, pass the full response to the finished callback.
        finished_callback(full_response)
        chat_history.add_chatbot_message(full_response)
    except Exception as e:
        error_message = f"\n[Error: {str(e)}]"
        update_callback(error_message)
        finished_callback("")



class ChatHistory:
    def __init__(self):
        # Stores messages as a list of dictionaries.
        self.history = []

    def add_user_message(self, message):
        """Add a user's message to the conversation history."""
        self.history.append({"role": "user", "content": message})

    def add_chatbot_message(self, message):
        """Add the chatbot's response to the conversation history."""
        self.history.append({"role": "assistant", "content": message})

    def get_conversation(self):
        """Return the entire conversation history."""
        return self.history

# TESTING

def update_callback(content):
    """Print streamed content as it arrives."""
    print(content, end="", flush=True)


def finished_callback(full_response):
    """Print the full response after streaming is complete."""
    pass

