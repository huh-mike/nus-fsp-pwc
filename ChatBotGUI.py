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
    conversation = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat test.")
            break

        # user_input -> gpt-text-embedding model to generate a embedding
        # for that embedding, find vector simiarlity in the database and fetch the actual contents as a part of system input.
        system_rag_context = None #result from above

        conversation.append({"role": "system", "content": f"Your reference for this question: {system_rag_context}"})
        conversation.append({"role": "user", "content": user_input})

        print("\nAssistant:", end=" ", flush=True)
        gpt_stream_responses(conversation, update_callback, finished_callback)

'''
1. 爬蟲軟件會從iras爬最新的文章
2. 有一個tagger会把每一个最新的文章赋予tag (classification) => 新的數據庫
- tagger to reconstruct with the following steps:
    - pre-process the crawled data with GPT4o (to-do: generate an universal prompt to get accurate summaries)
    - give summaries to text embedding model to generate better embeddings.
    - use vector similarity to match best k-tags
    - use gpt-4o-mini to validate the k-tags against the original summary.
    - lastly, add tags in dictionary.
3. 有一个chatbot會根據新的數據庫作為context來回答tax 相關問題

frameworks to consider: llamaindex, langchain.
'''