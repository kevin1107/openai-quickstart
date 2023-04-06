import openai
from openai import OpenAIError

import config

openai.api_key = config.ProductionConfig.SECRET_KEY


def generateChatResponse(prompt):
    messages = []
    messages.append({"role": "system",
                     "content": "You are a helpful assistant,Suggest three names for an animal that is a superhero."})
    messages.append({"role": "user",
                     "content": "Cat"})
    messages.append({"role": "assistant",
                     "content": "Captain Sharpclaw, Agent Fluffball, The Incredible Feline"})
    messages.append({"role": "user",
                     "content": "Dog"})
    messages.append({"role": "assistant",
                     "content": "Ruff the Protector, Wonder Canine, Sir Barks-a-Lot"})
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)
    try:
        response = openai.ChatCompletion.create(model=config.ProductionConfig.OPENAI_MODEL, messages=messages,
                                                timeout=int(config.ProductionConfig.OPENAI_TIMEOUT))
        answer = response['choices'][0]['message']['content']
    except OpenAIError as e:
        answer = f"request error：{e}"
    return answer
