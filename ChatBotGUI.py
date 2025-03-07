from GPTServices import gpt_stream_responses, gpt_generate_embedding, ChatHistory
from RAGServices import fetch_relevant_documents
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
    print("Tax Assistant Chatbot. Type 'exit' to stop.")
    print("This chatbot uses RAG to find relevant tax information for your queries.")

    # Initialise Conversation
    chat_history = ChatHistory()
    # Add system message to guide the assistant's behavior
    chat_history.history = [
        {"role": "system", "content": "You are a tax assistant AI that specializes in Singapore tax information. "
                                     "Provide concise, accurate answers based on the reference information provided. "
                                     "If you don't have specific information to answer a question, acknowledge this "
                                     "limitation and provide general information if possible."}
    ]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Exiting chat test.")
            break

        # Get relevant documents using RAG
        print("Searching for relevant tax information...")
        system_rag_context = fetch_relevant_documents(user_input)
        
        # Create a temporary conversation for this exchange
        conversation = chat_history.get_conversation().copy()
        # Add the RAG context as a system message
        conversation.append({"role": "system", "content": f"Your reference for this question: {system_rag_context}"})
        # Add the user's question
        conversation.append({"role": "user", "content": user_input})

        # Add the user message to the history
        chat_history.add_user_message(user_input)

        print("\nAssistant:", end=" ", flush=True)
        # Use the temporary conversation that includes the RAG context
        gpt_stream_responses(conversation, update_callback, finished_callback, chat_history)


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