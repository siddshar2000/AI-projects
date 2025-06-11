# GenAI based chat bot for helping programmer
This project is a GenAI based chatbot answering programming questions

## Installation
1.  Clone the repository: `git clone [repository URL]`
2.  Install dependencies: `pip install openai textwrap dotenv`
3.  Add .env file with key OPEN_AI_API_KEY, containing your OpenAI api key. [OpenAI site](https://platform.openai.com/settings/organization/api-keys)
4.  Run the project: `python main.py`

## Usage
This project is a GenAI based chatbot answering programming questions using OpenAI
This is a conversation based bot so you can ask questions based on previous answers. Validation of user asked question to be related to programming.
Max allowed questions are 10.

## Model Scorecard
### Input data
User entered programming questions only.

### Model used
OpenAI gpt-4o-mini

### Sample output 
**Sample questions and answer**
```
Some sample programming questions below, which will give idea for how to use this


Question number 1

        You asked: "What's a popular choice for a first programming language?"
        Response: 
             A popular choice for a first programming language is Python. It is widely recommended for beginners due to its simple and readable syntax, which allows newcomers to focus on learning programming concepts without getting overwhelmed by complex syntax rules. Additionally, Python has a large community, extensive libraries, and resources, making it easy to find help and tutorials. Other common choices for beginners include JavaScript, which is great for web development, and Scratch, which introduces programming concepts through visual block-based coding.
------------------------------------------------------------
```

**Validation of only programming questions allowed:**
```
Enter your next programming question or type "exit"
What is today date?
This is not a valid programming question, try again.

Enter your next programming question or type "exit"
```

**Conversation based, allowing question based on previous answers**
```
Enter your next programming question or type "exit"
Who is godfather of this language?


Question number 4

        You asked: "Who is godfather of this language?"
        Response: 
             The "godfather" of Python is Guido van Rossum. He created Python in the late 1980s and released the first version in 1991. Guido van Rossum served as the "Benevolent Dictator For Life" (BDFL) of Python, guiding its development and making decisions about its future until he stepped down from that role in July 2018. His vision and design philosophy have greatly shaped Python into the widely-used programming language it is today.
------------------------------------------------------------
```
