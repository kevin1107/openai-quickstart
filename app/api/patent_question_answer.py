import ast  # for converting embeddings saved as strings back to arrays

import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
from scipy import spatial  # for calculating vector similarities for search

embeddings_path = "D:\patsnap\embedding\patentBiblioInfo_with_embeddings_1k.csv"
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = ""

df = pd.read_csv(embeddings_path)
print(df.head(2))

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)


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


# examples
strings, relatednesses = strings_ranked_by_relatedness("whose Application Number is ES2014801674T", df, top_n=5)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)
