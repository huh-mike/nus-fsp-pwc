from GPTServices import gpt_stream_responses, gpt_generate_embedding, ChatHistory
from DataBasePipline import fetch_relevant_documents
import dotenv

dotenv.load_dotenv()
# Internal

def update_callback(content):
    print(content, end="", flush=True)


def finished_callback(full_response):
    print("\n\n[COMPLETE] Full Response:")
    print(full_response)

# External

def run_chatbotgui():
    print("Chat Streaming Test. Type 'exit' to stop.")

    # Initialise Conversation
    chat_history = ChatHistory()
    conversation = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat test.")
            break

        user_embedding = gpt_generate_embedding(user_input)
        print(f"Your Question's embedding is generated: {user_embedding}")

        # system_rag_context = fetch_relevant_documents(user_embedding)
        # print(f"Your reference for this question: {system_rag_context}")

        conversation.append({"role": "system", "content": "Your reference for this question: {system_rag_context}"})
        conversation.append({"role": "user", "content": user_input})

        print("\nAssistant:", end=" ", flush=True)
        gpt_stream_responses(conversation, update_callback, finished_callback,chat_history)

        # user_input -> gpt-text-embedding model to generate a embedding
        # for that embedding, find vector simiarlity in the database and fetch the actual contents as a part of system input.
if __name__ == "__main__":
    run_chatbotgui()

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