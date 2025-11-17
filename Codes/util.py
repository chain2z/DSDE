import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objs as go

def createNetwork_weighted(df_edges: pd.DataFrame , graph_title = "Network graph" , idx = "") -> None:
    """
    df_edges: must have 'from', 'to', optional 'weight'.
    If 'weight' is missing, edges are aggregated undirectly (A,B == B,A).
    """
    allAuthor = set(df_edges["to"].values) | set(df_edges["from"].values)

    # --- UI controls ---
    col1, col2, col3 = st.columns(3)
    with col1:
        layout_name = st.selectbox(
            "Choose a network layout",
            ("Random Layout", "Spring Layout", "Shell Layout", "Kamada Kawai Layout", "Spectral Layout"),
            key=f"Network graph selectbox1 {idx}"
        )
    with col2:
        color_name = st.selectbox(
            "Choose color of the nodes",
            ("Blue", "Red", "Green", "Orange", "Red-Blue", "Yellow-Green-Blue"), 
            key=f"Network graph selectbox2 {idx}"
        )
    with col3:
        authors = st.multiselect(label=f"Choose {idx}" , options=allAuthor,key=f"Network graph multiselect {idx}")

    df_edges = df_edges[df_edges["from"].isin(authors) | df_edges["to"].isin(authors)]
    
    # Build undirected weighted graph
    G = nx.from_pandas_edgelist(
        df_edges,
        source="from",
        target="to",
        edge_attr="weight",
        create_using=nx.Graph()
    )

    # Layouts (spring uses weights)
    layout_funcs = {
        "Random Layout": lambda g: nx.random_layout(g),
        "Spring Layout": lambda g: nx.spring_layout(g, k=0.5, iterations=50, weight="weight"),
        "Shell Layout": lambda g: nx.shell_layout(g),
        "Kamada Kawai Layout": lambda g: nx.kamada_kawai_layout(g),
        "Spectral Layout": lambda g: nx.spectral_layout(g),
    }
    pos = layout_funcs[layout_name](G)

    # Colorscales
    colorscale_map = {
        "Blue": "Blues",
        "Red": "Reds",
        "Green": "Greens",
        "Orange": "Oranges",
        "Red-Blue": "RdBu",
        "Yellow-Green-Blue": "YlGnBu",
    }
    colorscale = colorscale_map[color_name]

    # --- Edge trace ---
    edge_x = []
    edge_y = []
    for u, v, w in G.edges(data="weight", default=1.0):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # --- Node positions ---
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Weighted degree (sum of edge weights for each node)
    degrees = dict(G.degree(weight="weight"))
    node_color = [degrees[node] for node in G.nodes()]
    node_text = [
        f"{node} # weighted connections: {degrees[node]}"
        for node in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_text,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            color=node_color,
            size=20,
            # ↓↓↓ simplified colorbar: only properties supported on older plotly
            colorbar=dict(
                thickness=10,
                title="Weighted degree",
            ),
            line=dict(width=0),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=graph_title,
            title_x=0.45,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)
def get_classification_report(d):
    report_df = pd.DataFrame(d).transpose()
    report_df = report_df.round(3)

    report_df.loc["accuracy",["precision","recall", "support"]] = ["", "" , report_df.loc["macro avg" , "support"]]
    st.table(report_df)