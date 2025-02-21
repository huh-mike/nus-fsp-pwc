from GPTServices import gpt_stream_responses

# Internal

def update_callback(content):
    print(content, end="", flush=True)


def finished_callback(full_response):
    print("\n\n[COMPLETE] Full Response:")
    print(full_response)

    # To update for response validation

# External

def run_chatbotgui():
    print("Chat Streaming Test. Type 'exit' to stop.")

    # Initialise Conversation
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat test.")
            break

        conversation.append({"role": "user", "content": user_input})

        print("\nAssistant:", end=" ", flush=True)
        gpt_stream_responses(conversation, update_callback, finished_callback)

