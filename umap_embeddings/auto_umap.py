import numpy as np
import plotly.graph_objects as go
import umap
from tqdm import tqdm

# pip install umap-learn


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


def make_umap(
    vector_db, retrieved_embeddings, query_embedding, query, session_number
) -> str:
    embeddings = vector_db.get(include=["embeddings", "documents"])
    umap_transform = umap.UMAP(n_components=2).fit(embeddings["embeddings"])
    projected_dataset_embeddings = project_embeddings(
        embeddings["embeddings"], umap_transform
    )
    projected_query_embedding = project_embeddings([query_embedding], umap_transform)
    projected_retrieved_embeddings = project_embeddings(
        retrieved_embeddings, umap_transform
    )
    # Scatter plot for database embeddings
    scatter_dataset = go.Scatter(
        x=projected_dataset_embeddings[:, 0],
        y=projected_dataset_embeddings[:, 1],
        mode="markers",
        marker=dict(size=10, color="blue"),
        name="Dataset Embeddings",
        hoverlabel=dict(font=dict(color="white")),
    )

    # Scatter plot for query embedding
    scatter_query = go.Scatter(
        x=[projected_query_embedding[0, 0]],
        y=[projected_query_embedding[0, 1]],
        mode="markers",
        marker={"size": 10, "color": "red", "symbol": "x"},
        name="Query Embedding",
    )

    # Scatter plot for retrieved embeddings
    scatter_retrieved = go.Scatter(
        x=projected_retrieved_embeddings[:, 0],
        y=projected_retrieved_embeddings[:, 1],
        mode="markers",
        marker=dict(size=15, color="green", symbol="circle"),
        name="Retrieved Embeddings",
    )

    # Define layout
    layout = go.Layout(
        title=query, xaxis=dict(visible=False), yaxis=dict(visible=False)
    )

    # Create figure
    fig = go.Figure(
        data=[scatter_dataset, scatter_query, scatter_retrieved], layout=layout
    )
    path = f"umap_embeddings/results/{session_number}.png"
    fig.write_image(path)
    return path
