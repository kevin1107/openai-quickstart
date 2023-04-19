import ast  # for converting embeddings saved as strings back to arrays
import os

import openai
import pandas as pd
from flasgger import swag_from
from flask import jsonify, request
from openai.error import OpenAIError
from scipy import spatial  # for calculating vector similarities for search

from . import api

PATH = lambda p: os.path.abspath(os.path.join(os.path.dirname(__file__), p))

EMBEDDING_MODEL = "text-embedding-ada-002"

cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(cur_path)
# Get the path where the project is located, if you add the directory name, switch to the directory
config_path = os.path.join(cur_path, 'data')
print(config_path)
embedding_cache_path = PATH(config_path + r'\patentBiblioInfo_with_embeddings_1k.csv')
df = pd.read_csv(embedding_cache_path)

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)


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
        openai.api_key = secret_key
        top_results = []
        strings, relatednesses = strings_ranked_by_relatedness(question, df, top_n=top_n)
        for string, relatedness in zip(strings, relatednesses):
            print(f"{relatedness=:.3f}")
            print(string)
            top_results.append(f"{relatedness=:.3f}")
            top_results.append(string)
    except OpenAIError:
        print(OpenAIError)
        return {'code': 'error'}, 500
    return jsonify(top_results), 200


# search function
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
        (row["combined"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]
