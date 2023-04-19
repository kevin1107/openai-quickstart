import os
import pickle

import numpy as np
import openai
import pandas as pd
from flasgger import swag_from
from flask import jsonify, request
from openai.embeddings_utils import (
    cosine_similarity,
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
from openai.error import OpenAIError
from sklearn.cluster import KMeans

from . import api

PATH = lambda p: os.path.abspath(os.path.join(os.path.dirname(__file__), p))

EMBEDDING_MODEL = "text-embedding-ada-002"

cur_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(cur_path)
# Get the path where the project is located, if you add the directory name, switch to the directory
config_path = os.path.join(cur_path, 'data')
print(config_path)
embedding_cache_path = PATH(config_path + r'\recommendations_embeddings_cache.pkl')

# load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


@api.route('/embeddings', methods=['POST'])
@swag_from('embeddings_specs.yml')
def get_embeddings():
    try:
        embedding_info = request.get_json()
        if not bool(embedding_info):
            return {'code': 'error', 'msg': 'request body is not null!'}, 400
        secret_key = embedding_info['secretKey']
        text = embedding_info['text']
        if secret_key is None:
            return {'code': 'error', 'msg': 'secretKey is not null!'}, 400
        if text is None:
            return {'code': 'error', 'msg': 'text is not null!'}, 400
        openai.api_key = secret_key
        response = openai.Embedding.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        output = response['data'][0]['embedding']
    except OpenAIError:
        print(OpenAIError)
        return {'code': 'error'}, 500
    return jsonify(output), 200


@api.route('/embeddings/zero_shot', methods=['POST'])
@swag_from('embeddings_zero_shot_specs.yml')
def evaluate_embeddings_approach():
    try:
        embedding_info = request.get_json()
        if not bool(embedding_info):
            return {'code': 'error', 'msg': 'request body is not null!'}, 400
        secret_key = embedding_info['secretKey']
        label1 = embedding_info['label1']
        label2 = embedding_info['label2']
        if secret_key is None:
            return {'code': 'error', 'msg': 'secretKey is not null!'}, 400
        if label1 is None:
            return {'code': 'error', 'msg': 'label1 is not null!'}, 400
        if label2 is None:
            return {'code': 'error', 'msg': 'label2 is not null!'}, 400
        openai.api_key = secret_key
        embedding1 = get_embedding(label1)
        embedding2 = get_embedding(label2)
        similarity = cosine_similarity(embedding1, embedding2)
    except OpenAIError:
        print(OpenAIError)
        return {'code': 'error'}, 500
    return jsonify({"cosine_similarity": similarity}), 200


@api.route('/embeddings/clustering', methods=['POST'])
@swag_from('embeddings_clustering_specs.yml')
def kmeans_clusters():
    try:
        embedding_info = request.get_json()
        if not bool(embedding_info):
            return {'code': 'error', 'msg': 'request body is not null!'}, 400
        secret_key = embedding_info['secretKey']
        texts = embedding_info['texts']
        if secret_key is None:
            return {'code': 'error', 'msg': 'secretKey is not null!'}, 400
        if texts is None:
            return {'code': 'error', 'msg': 'texts is not null!'}, 400
        output = []
        embeddings = np.array([embeddings_text(secret_key, text) for text in texts])
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(embeddings)
        for i, cluster in enumerate(clusters):
            output.append(f"Text: {texts[i]} - Cluster: {cluster}")
    except OpenAIError:
        print(OpenAIError)
        return {'code': 'error'}, 500
    return jsonify(output), 200


@api.route('/embeddings/recommendation', methods=['POST'])
@swag_from('embeddings_recommendation_specs.yml')
def recommendations_from_strings():
    try:
        embedding_info = request.get_json()
        if not bool(embedding_info):
            return {'code': 'error', 'msg': 'request body is not null!'}, 400
        secret_key = embedding_info['secretKey']
        descriptions = embedding_info['descriptions']
        if secret_key is None:
            return {'code': 'error', 'msg': 'secretKey is not null!'}, 400
        if descriptions is None:
            return {'code': 'error', 'msg': 'descriptions is not null!'}, 400
        openai.api_key = secret_key
        article_descriptions = descriptions
        recommendations_articles = get_recommendations_from_strings(
            strings=article_descriptions,  # let's base similarity off of the article description
            index_of_source_string=0,  # let's look at articles similar to the first one about Tony Blair
            k_nearest_neighbors=5,  # let's look at the 5 most similar articles
        )
    except OpenAIError:
        print(OpenAIError)
        return {'code': 'error'}, 500
    return jsonify(recommendations_articles), 200


def get_recommendations_from_strings(
        strings: list[str],
        index_of_source_string: int,
        k_nearest_neighbors: int = 1,
        model=EMBEDDING_MODEL,
) -> list[str]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    embeddings = [embedding_from_string(string, model=model) for string in strings]
    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # print out source string
    query_string = strings[index_of_source_string]
    print(f"Source string: {query_string}")
    # print out its k nearest neighbors
    k_counter = 0
    recommendations = []
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )
        recommendations.append(f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}""")
    return recommendations


# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
        string: str,
        model: str = EMBEDDING_MODEL,
        embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


def embeddings_text(key, text):
    openai.api_key = key
    response = openai.Embedding.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    embeddings = response['data'][0]['embedding']
    return embeddings
