from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            )

def create_gpt_responses(prompt):
    response = openai_client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": 'You are a helpful agent'},
            {"role": "user", "content": prompt}]
    )
    response_content = response.choices[0].message.content
    return response_content


if __name__ == "__main__":
    print(create_gpt_responses("Say something funny"))
