from ChatBotGUI import run_chatbotgui


def main():
    """
    Main entry point for the Tax Assistant application.
    Ensures all necessary files are available before running the chatbot.
    """
    try:
        # Import necessary modules (will raise ImportError if any are missing)
        import pandas as pd
        import numpy as np
        from openai import OpenAI
        
        # Run the chatbot GUI
        run_chatbotgui()
    except ImportError as e:
        print(f"Error: Missing required package. {e}")
        print("Please install required packages using: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()