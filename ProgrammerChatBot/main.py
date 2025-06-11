from openai import OpenAI
from textwrap import indent
from dotenv import load_dotenv
import os

load_dotenv()

MODEL="gpt-4o-mini"
API_KEY = os.environ['OPEN_AI_API_KEY']
MAX_QUESTIONS = 10

client = OpenAI(api_key=API_KEY)

# Ask GenAI to role play as programming expert
history = [
    {
        "role": "system",
        "content": "You are a programming expert. Answer questions about programming languages"
    }
]

sample_qestions = [
    "What's a popular choice for a first programming language?",
    "What are some advantages of learning it as my first language?",
    "Can you show me a simple 'Hello World' program written in that language?",
]
num_q = 0

def print_qa(num_q, question, resp_message):
    indent_resp = indent(resp_message.content, '    ')
    print(f"\n\nQuestion number {num_q}\n")
    print(f"\tYou asked: \"{question}\"")
    print(f"\tResponse: \n\t {indent_resp}")
    print("-"*60)

def process_question(num_q, question):
    history.append({
        "role": "user",
        "content": question
        })
    response = client.chat.completions.create(model=MODEL, messages=history)
    history.append(response.choices[0].message)
    print_qa(num_q, question, response.choices[0].message)

def validate_question(question):
    response = client.chat.completions.create(
        model=MODEL,
        messages = [
            {
                "role": "user",
                "content":  f"Is this is a programing question: \"{question}\". Response in yes or no only. Without punctuation."
            }
        ]
    )

    return response.choices[0].message.content.strip().lower() == 'yes'
    
# Start with showing sample programming questions
print("Some sample programming questions below, which will give idea for how to use this") 
for question in sample_qestions:
    num_q += 1
    process_question(num_q, question)


print("\n\tNow you ask programming questions!")

while(True):
    if (num_q >= MAX_QUESTIONS):
        print("Reach maximum questions you can ask. Bye!")
        exit(0)

    print("\nEnter your next programming question or type \"exit\"")

    user_q = str(input())
    if (user_q.lower() == 'exit'):
        print("Bye!")
        exit(0)

    if (validate_question(user_q) == False):
        print("This is not a valid programming question, try again.")
        continue

    num_q +=1 
    process_question(num_q, user_q)