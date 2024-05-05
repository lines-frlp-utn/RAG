def test_plotly():
    import plotly.graph_objects as go

    scatter_retrieved = go.Scatter(
        x=[0, 1],
        y=[4, 2],
        mode="markers",
        marker=dict(size=15, color="green", symbol="circle"),
        name="Retrieved Embeddings",
    )

    # Define layout
    layout = go.Layout(
        title="test", xaxis=dict(visible=False), yaxis=dict(visible=False)
    )

    # Create figure
    fig = go.Figure(data=[scatter_retrieved], layout=layout)
    path = f"app/umap_results/1.png"
    fig.write_image(path)
