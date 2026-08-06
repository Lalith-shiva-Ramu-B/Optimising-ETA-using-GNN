from __future__ import annotations
from typing import Iterable, Optional, Set
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# KPI gauges (kept from the original streamlit_app.py, lightly cleaned up)
def kpi_indicator(label: str, value: float, prefix: str = "", suffix: str = ""):
    """Single-number KPI card (the original ``plot_metrics`` helper)."""
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=value,
            number={"prefix": prefix, "suffix": suffix, "font": {"size": 28}},
            title={"text": label, "font": {"size": 16}},
        )
    )
    fig.update_layout(height=120, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# Network graph
def plot_network(
    graph: nx.DiGraph,
    breached_edges: Optional[Iterable[tuple]] = None,
    bottleneck_nodes: Optional[Set[str]] = None,
    layout_seed: int = 42,
    max_nodes: int = 300,
) :
    
    breached_edges = set(breached_edges or [])
    bottleneck_nodes = set(bottleneck_nodes or [])

    display_graph = graph
    if graph.number_of_nodes() > max_nodes:
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        display_graph = graph.subgraph(top_nodes).copy()

    pos = nx.spring_layout(display_graph, seed=layout_seed)
    degrees = dict(display_graph.degree())
    max_degree = max(degrees.values()) if degrees else 1

    fig = go.Figure()

    # Edges: draw breached and normal separately so each group gets one color/width rule.
    for is_breach, color in ((True, "crimson"), (False, "rgba(150,150,150,0.5)")):
        edge_x, edge_y = [], []
        for u, v, data in display_graph.edges(data=True):
            if ((u, v) in breached_edges) != is_breach:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        if edge_x:
            width = 2.5 if is_breach else 1
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="none",
                    name="Breached corridor" if is_breach else "Normal corridor",
                )
            )

    # Nodes
    node_x = [pos[n][0] for n in display_graph.nodes()]
    node_y = [pos[n][1] for n in display_graph.nodes()]
    node_size = [8 + 22 * (degrees[n] / max_degree) for n in display_graph.nodes()]
    node_color = ["orange" if n in bottleneck_nodes else "#1f77b4" for n in display_graph.nodes()]
    node_text = [f"{n}<br>degree: {degrees[n]}" for n in display_graph.nodes()]

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            text=node_text, hoverinfo="text", name="Facility",
        )
    )

    fig.update_layout(
        showlegend=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=650, margin=dict(l=10, r=10, t=30, b=10),
        title="Logistics Network (node size = degree, orange = bottleneck hub, red = breached lane)",
    )
    return fig


# Bottleneck / hub metrics
def plot_bottleneck_bar(hub_metrics_df: pd.DataFrame, metric: str, top_n: int = 15):
    """Horizontal bar chart of the top-N facilities for a given hub metric
    (e.g. Betweenness, In_Degree, Out_Degree, Clustering, SLA_Breach_Contribution_%)."""
    top_df = hub_metrics_df.sort_values(by=metric, ascending=False).head(top_n)
    fig = px.bar(
        top_df.sort_values(by=metric), x=metric, y="Facility", orientation="h",
        title=f"Top {top_n} facilities by {metric}",
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# Corridor analysis
def plot_delay_ratio_distribution(corridor_df: pd.DataFrame):
    fig = px.histogram(
        corridor_df, x="median_delay_ratio", nbins=40,
        title="Distribution of Median Delay Ratio Across Corridors",
    )
    fig.add_vline(x=1.2, line_dash="dash", line_color="red", annotation_text="20% SLA threshold")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_route_type_comparison(corridor_df: pd.DataFrame):
    summary = corridor_df.groupby("route_type").agg(
        avg_delay_ratio=("median_delay_ratio", "mean"),
        total_breaches=("breaches", "sum"),
        total_trips=("total_trips", "sum"),
    ).reset_index()
    fig = px.bar(
        summary, x="route_type", y="avg_delay_ratio", color="route_type",
        title="Average Delay Ratio by Route Type",
        hover_data=["total_breaches", "total_trips"],
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig

def plot_route_type_breaches(breached_df: pd.DataFrame):
    summary = breached_df.groupby("route_type").agg(
         avg_delay_ratio=("median_delay_ratio", "mean"),
        total_breaches=("breaches", "sum"),
        total_trips=("total_trips", "sum"),
    ).reset_index()
    fig = px.bar(
        summary, x="route_type", y="total_breaches", color="route_type",
        title="Total Breaches by Route Type",
        hover_data=["total_trips"],
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig
def plot_time_of_day_comparison(corridor_df: pd.DataFrame):
    bucket_labels = {0: "Night", 1: "Morning", 2: "Afternoon", 3: "Evening"}
    summary = corridor_df.groupby("time_of_day").agg(
        avg_delay_ratio=("median_delay_ratio", "mean"),
        total_breaches=("breaches", "sum"),
    ).reset_index()
    summary["time_of_day_label"] = summary["time_of_day"].map(bucket_labels)
    fig = px.bar(
        summary, x="time_of_day_label", y="avg_delay_ratio",
        title="Average Delay Ratio by Time of Day",
        hover_data=["total_breaches"],
        category_orders={"time_of_day_label": ["Night", "Morning", "Afternoon", "Evening"]},
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_time_of_day_breaches(breached_df: pd.DataFrame):
    bucket_labels = {0: "Night", 1: "Morning", 2: "Afternoon", 3: "Evening"}
    summary = breached_df.groupby("time_of_day").agg(
        avg_delay_ratio=("median_delay_ratio", "mean"),
        total_breaches=("breaches", "sum"),
    ).reset_index()
    summary["time_of_day_label"] = summary["time_of_day"].map(bucket_labels)
    fig = px.bar(
        summary, x="time_of_day_label", y="total_breaches",
        title="Total breaches by Time of Day",
        hover_data=["avg_delay_ratio"],
        category_orders={"time_of_day_label": ["Night", "Morning", "Afternoon", "Evening"]},
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ML model evaluation
def plot_model_comparison_bar(base_mae: float, graph_mae: float, base_acc: float, graph_acc: float,title):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Random Forest (baseline)", x=["MAE (mins)", "Within-15% Accuracy (%)"],
                          y=[base_mae, base_acc]))
    fig.add_trace(go.Bar(name="GraphSAGE", x=["MAE (mins)", "Within-15% Accuracy (%)"],
                          y=[graph_mae, graph_acc]))
    fig.update_layout(barmode="group", height=400, title=title,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    fig = px.scatter(
        x=y_true, y=y_pred, opacity=0.5,
        labels={"x": "Actual segment time (mins)", "y": "Predicted segment time (mins)"},
        title=title,
    )
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                              line=dict(dash="dash", color="red")))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_error_histogram(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    errors = np.asarray(y_pred) - np.asarray(y_true)
    fig = px.histogram(x=errors, nbins=50, title=title, labels={"x": "Prediction error (mins)"})
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig
