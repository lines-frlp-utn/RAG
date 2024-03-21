import chromadb
import numpy as np
import plotly.graph_objects as go
import umap
from chromadb.config import Settings
from embedding_model import get_sentence_embedding
from splitter import text_splitter
from tqdm import tqdm

# pip install umap-learn
from vector_db import store_embedding_with_document

db = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
collection = db.get_or_create_collection(name="make_umap")
with open("umap_embeddings/document.txt") as file:
    text = file.read()

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    chunk = chunk.replace("\n", " ")
    embedding = get_sentence_embedding(chunk)
    store_embedding_with_document(embedding, str(i), chunk, collection=collection)

embeddings = collection.get(include=["embeddings", "documents"])
np_embeddings = np.array(embeddings["embeddings"])

umap_transform = umap.UMAP(n_components=4).fit(np_embeddings)


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 4))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


projected_dataset_embeddings = project_embeddings(
    embeddings["embeddings"], umap_transform
)

query = "What is overwatch 2?"

results = collection.query(
    query_texts=query, n_results=3, include=["documents", "embeddings"]
)

query_embedding = get_sentence_embedding([query])
retrieved_embeddings = results["embeddings"][0]

projected_query_embedding = project_embeddings([query_embedding], umap_transform)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

db_documents = []
for x in embeddings["documents"]:
    result = ""
    for i in range(0, len(x), 70):
        result += x[i : i + 70] + "<br>"
    db_documents.append(result)

retrieved_documents = []
for x in results["documents"][0]:
    result = ""
    for i in range(0, len(x), 70):
        result += x[i : i + 70] + "<br>"
    retrieved_documents.append(result)

# Scatter plot for database embeddings
scatter_dataset = go.Scatter3d(
    x=projected_dataset_embeddings[:, 0],
    y=projected_dataset_embeddings[:, 1],
    z=projected_dataset_embeddings[:, 2],
    mode="markers",
    marker=dict(size=10, color=projected_dataset_embeddings[:, 3]),
    name="Dataset Embeddings",
    hovertext=db_documents,
    hoverinfo="text",
    hoverlabel=dict(font=dict(color="white")),
)

# Scatter plot for query embedding
scatter_query = go.Scatter3d(
    x=[projected_query_embedding[0, 0]],
    y=[projected_query_embedding[0, 1]],
    z=[projected_query_embedding[0, 2]],
    mode="markers",
    marker={"size": 10, "color": "red", "symbol": "x"},
    name="Query Embedding",
    hovertext=query,
    hoverinfo="text",
)

# Scatter plot for retrieved embeddings
scatter_retrieved = go.Scatter3d(
    x=projected_retrieved_embeddings[:, 0],
    y=projected_retrieved_embeddings[:, 1],
    z=projected_retrieved_embeddings[:, 2],
    mode="markers",
    marker=dict(size=15, color="green", symbol="circle"),
    name="Retrieved Embeddings",
    hovertext=retrieved_documents,
    hoverinfo="text",
)

# Define layout
layout = go.Layout(title=query, xaxis=dict(visible=False), yaxis=dict(visible=False))

# Create figure
fig = go.Figure(data=[scatter_dataset, scatter_query, scatter_retrieved], layout=layout)

fig.write_html("ow.html")
db.delete_collection(name="make_umap")
