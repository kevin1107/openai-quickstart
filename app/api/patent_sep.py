import ast
import concurrent.futures
import concurrent.futures
import datetime
import os

import openai
import pandas as pd
import tiktoken
from flasgger import swag_from
from flask import jsonify, request
from openai.error import OpenAIError
from scipy import spatial

from . import api

# 使用你的 API 密钥
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

PATH = lambda p: os.path.abspath(os.path.join(os.path.dirname(__file__), p))

cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(cur_path)
# Get the path where the project is located, if you add the directory name, switch to the directory
config_path = os.path.join(cur_path, 'data')
print(config_path)
# embedding_cache_path = PATH(config_path + r'\patentBiblioInfo_with_embeddings_1k.csv')
now = datetime.datetime.now()
print("embedding开始加载时间是：" + now.strftime("%Y-%m-%d %H:%M:%S"))

# 本地调试
# embedding_cache_path = PATH(config_path + r'/embedding_res.csv')
# df = pd.read_csv(embedding_cache_path)
# 线上调试
df = pd.read_csv('/home/openai/app/data/embedding_res.csv')
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)
now = datetime.datetime.now()
print("embedding加载完成时间是：" + now.strftime("%Y-%m-%d %H:%M:%S"))


def convert_embedding(embedding_str):
    return ast.literal_eval(embedding_str)


def read_embedding_cache(embedding_cache_path):
    df = pd.read_csv(embedding_cache_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        df['embedding'] = list(executor.map(convert_embedding, df['embedding']))
    return df


def ask(
        query: str,
        df: pd.DataFrame = df,
        model: str = GPT_MODEL,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
        top_n: int = 100
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    strings, relatednesses = strings_ranked_by_relatedness_parallel(query, df, top_n=top_n)
    message = query_message(query, strings, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the patent information."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]

    result = {
        "question": query,
        "similar_patents": [{"relatedness": relatedness, "patent": string} for relatedness, string in
                            zip(relatednesses[:20], strings[:20])],
        "gpt_answer": response_message
    }

    return result


# 通过embeddings和spatial.distance.cosine算法，获取与问题最相关的几条文本
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["input"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def process_row(row, query_embedding, relatedness_fn):
    return (row["input"], relatedness_fn(query_embedding, row["embedding"]))


# 多线程比较embeddings
def strings_ranked_by_relatedness_parallel(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]

    strings_and_relatednesses = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, row in df.iterrows():
            future = executor.submit(process_row, row, query_embedding, relatedness_fn)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            strings_and_relatednesses.append(result)

    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


# 计算给定字符串的 token 数量
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# 将问题与相关文本组合成一个prompt
def query_message(
        query: str,
        similar_text: str,
        model: str,
        token_budget: int
) -> str:
    introduction = 'Use the following articles on patent information to answer follow-up questions. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in similar_text:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


@api.route('/patent/sep', methods=['POST'])
@swag_from('patent_sep_specs.yml')
def patent_sep():
    try:
        embedding_info = request.get_json()
        if not bool(embedding_info):
            return {'code': 'error', 'msg': 'request body is not null!'}, 400
        secret_key = embedding_info['secretKey']
        question = embedding_info['question']
        top_n = embedding_info['top_n']
        if top_n > 20:
            top_n = 20
        openai.api_key = secret_key
        top_results = []
        result_dict = ask(query=question, df=df, top_n=top_n)
        question = result_dict["question"]
        similar_patents = result_dict["similar_patents"]
        gpt_answer = result_dict["gpt_answer"]
        # 打印结果
        print("Question:", question)
        top_results.append("Your Question: " + question)
        print("--------------------------")
        top_results.append(
            "=========================================================================================================")
        print("Similar patents: ")
        top_results.append("Similar patents: ")
        for patent in similar_patents:
            print(f"Relatedness: {patent['relatedness']:.3f}, Patent: {patent['patent']}")
            top_results.append(f"Relatedness: {patent['relatedness']:.3f}, Patent: {patent['patent']}")
        print("--------------------------")
        top_results.append(
            "=========================================================================================================")
        print("GPT's answer:", gpt_answer)
        top_results.append("GPT's answer: " + gpt_answer)

    except OpenAIError:
        print(OpenAIError)
        return {'code': 'error'}, 500
    return jsonify(top_results), 200
