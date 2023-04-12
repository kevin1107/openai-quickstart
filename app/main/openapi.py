import openai
from openai.error import OpenAIError

import config

openai.api_key = config.ProductionConfig.SECRET_KEY


def generateChatResponse(prompt):
    messages = []
    messages.append({"role": "system", "content": "Your name is AI_Rebot. You are a helpful assistant."})
    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)
    try:
        response = openai.ChatCompletion.create(model=config.ProductionConfig.OPENAI_MODEL, messages=messages,
                                                timeout=int(config.ProductionConfig.OPENAI_TIMEOUT))
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except OpenAIError as e:
        answer = f"request error：{e}"
    return answer
