import torch
import torch.nn as nn
import torch.nn.functional as F


class UMAP(nn.Module):
    def __init__(
        self,
        n_components,
        n_neighbors=15,
        min_dist=0.1,
        negative_sample_rate=5,
        init="spectral",
        device="cuda",
    ):
        super(UMAP, self).__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.negative_sample_rate = negative_sample_rate
        self.init = init
        self.device = device

    def forward(self, X):
        X = X.to(self.device)
        X = F.normalize(X, p=2, dim=1)  # Normalize input data

        # Initialize low-dimensional embeddings
        init_embedding = self.initialize_embedding(X.shape[0], X)

        # Perform optimization
        embedding = self.optimize(X, init_embedding)

        return embedding

    def initialize_embedding(self, n_samples, X):
        if self.init == "spectral":
            # Use spectral embedding for initialization
            from sklearn.manifold import SpectralEmbedding

            spec_emb = SpectralEmbedding(
                n_components=self.n_components, n_neighbors=self.n_neighbors
            )
            init_embedding = torch.Tensor(spec_emb.fit_transform(X.cpu().numpy()))
        else:
            # Random initialization
            init_embedding = torch.randn(
                n_samples, self.n_components, device=self.device
            )

        return init_embedding

    def optimize(self, X, init_embedding):
        # Define negative sampling
        n_neg_samples = X.shape[0] * self.negative_sample_rate
        neg_samples = torch.randn(n_neg_samples, self.n_components, device=self.device)

        # Perform optimization (not implemented, just for demonstration)
        # Placeholder for optimization process, you can replace this with actual optimization algorithm
        embedding = init_embedding.clone().detach()

        return embedding


umap = UMAP(n_components=2)
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

embeddings = collection.get(include=["embeddings", "documents"])["embeddings"]
torch_embeddings = torch.Tensor(embeddings)
# Crear instancia de UMAP
umap_model = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    negative_sample_rate=5,
    init="spectral",
)

# Mover el modelo a GPU
umap_model.to("cuda")

# Reducción de dimensionalidad
umap_embedding = umap_model(torch_embeddings)

scatter_dataset = go.Scatter(
    x=umap_embedding[:, 0],
    y=umap_embedding[:, 1],
    mode="markers",
    marker=dict(size=10),
    name="Dataset Embeddings",
    hoverinfo="text",
    hoverlabel=dict(font=dict(color="white")),
)
