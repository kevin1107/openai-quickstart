# imports
import openai
import pandas as pd
import tiktoken

from openai.embeddings_utils import get_embedding

openai.api_key = ""
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# load & inspect dataset
# input_datapath = "data/fine_food_reviews_1k.csv"  # to save space, we provide a pre-filtered dataset
input_datapath = "D:\patsnap\embedding\Reviews.csv"

df = pd.read_csv(input_datapath, index_col=0, encoding='ISO-8859-1', sep=',')
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
        "Content: " + df.Text.str.strip()
)
print(df.head(2))
print(len(df))

# subsample to 1k most recent reviews and remove samples that are too long
top_n = 1000
df = df.sort_values("Time").tail(
    top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens].tail(top_n)
len(df)

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

# This may take a few minutes
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))

df.to_csv("D:\patsnap\embedding\patentBiblioInfo_with_embeddings_1k.csv")
