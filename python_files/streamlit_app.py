import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from main import *
from plots import (
    kpi_indicator,
    plot_bottleneck_bar,
    plot_delay_ratio_distribution,
    plot_error_histogram,
    plot_model_comparison_bar,
    plot_network,
    plot_pred_vs_actual,
    plot_route_type_comparison,
    plot_time_of_day_comparison,plot_route_type_breaches,plot_time_of_day_breaches
)
st.set_page_config(page_title='ETA model Dashboard',layout='wide')

st.title("ETA Dashboard")
def load_data(path):
    data= pd.read_csv(path)
    return data

upload_file=st.sidebar.file_uploader("Choose the file")
if upload_file is None:
    st.info("upload a file ")
    st.stop()

df=load_data(upload_file)
page = st.sidebar.radio(
    "Dashboard section",
    (
        "Executive Dashboard",
        "Network Analysis",
        "Corridor Analysis",
        "Bottleneck Analysis",
        "ML Model",
        "Decision Framework",
    ),
)
with st.expander("Data preview"):
    st.dataframe(df)
corridor_df=corridor_setup(df)
breached_df=breached_find(corridor_df)
critical_hubs, hub_metrics_df=critical_hub_find(corridor_df)

if page == "Executive Dashboard":
        st.subheader("Key Performance Indicators")
        column_1, column_2, column_3, column_4, column_5 =st.columns(5)
        with column_1:
               st.plotly_chart(kpi_indicator("Total Trips", len(df)), use_container_width=True)
        with column_2:
               st.plotly_chart(kpi_indicator("Total Hubs", df["source_center"].nunique()), use_container_width=True)
        with column_3:
               st.plotly_chart(kpi_indicator("Total Corridors", len(corridor_df)), use_container_width=True)
        with column_4:
                st.plotly_chart(kpi_indicator("Breached Corridors", len(breached_df)), use_container_width=True)
        with column_5:
               st.plotly_chart(
            kpi_indicator("Avg Delay Ratio", float(np.mean(corridor_df["median_delay_ratio"])), suffix="x"),
            use_container_width=True)

elif page == "Network Analysis":
    st.subheader("Interactive Corridor Network")
    graph = build_graph(corridor_df)
    breached_edges = set(zip(breached_df["source_center"], breached_df["destination_center"]))
    fig = plot_network(graph, breached_edges=breached_edges, bottleneck_nodes=critical_hubs)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hub Metrics")
    st.caption(f"Identified {len(critical_hubs)} structurally critical (bottleneck) hubs.")
    st.dataframe(hub_metrics_df.sort_values(by="Betweenness", ascending=False), use_container_width=True)

elif page == "Corridor Analysis":
    st.subheader("Top Delayed Corridors")
    top_delayed = corridor_df.sort_values(by="median_delay_ratio", ascending=False).head(15)
    st.dataframe(top_delayed, use_container_width=True)

    st.subheader("Top Breached Corridors")
    st.dataframe(breached_df.head(15), use_container_width=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        st.plotly_chart(plot_delay_ratio_distribution(corridor_df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_route_type_comparison(corridor_df), use_container_width=True)
    with col3:
        st.plotly_chart(plot_route_type_breaches(breached_df), use_container_width=True)


    st.plotly_chart(plot_time_of_day_breaches(breached_df), use_container_width=True)

    st.plotly_chart(plot_time_of_day_comparison(corridor_df), use_container_width=True)

    st.subheader("Search Corridors")
    search_term = st.text_input("Filter by source or destination facility code")
    filtered = corridor_df
    if search_term:
        mask = (
            corridor_df["source_center"].str.contains(search_term, case=False, na=False)
            | corridor_df["destination_center"].str.contains(search_term, case=False, na=False)
        )
        filtered = corridor_df[mask]
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered corridors as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_corridors.csv",
        mime="text/csv",
    )
elif page == "Bottleneck Analysis":
    st.subheader("Top Bottleneck Hubs")
    st.dataframe(
        hub_metrics_df.sort_values(by="SLA_Breach_Contribution_%", ascending=False).head(20),
        use_container_width=True,
    )

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "Betweenness"), use_container_width=True)
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "In_Degree"), use_container_width=True)
    with metric_cols[1]:
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "Out_Degree"), use_container_width=True)
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "SLA_Breach_Contribution_%"), use_container_width=True)

    st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "Clustering"), use_container_width=True)

elif page == "ML Model":
        st.subheader("Baseline (Random Forest) vs. GraphSAGE")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Baseline MAE (min)", f"{metrics['base_mae']:.2f}")
        col2.metric("GraphSAGE MAE (min)", f"{metrics['graph_mae']:.2f}")
        col3.metric("Baseline within-15% accuracy", f"{metrics['base_acc']:.1f}%")
        col4.metric("GraphSAGE within-15% accuracy", f"{metrics['graph_acc']:.1f}%")

        st.plotly_chart(
            plot_model_comparison_bar(
                metrics["base_mae"], metrics["graph_mae"], metrics["base_acc"], metrics["graph_acc"]
            ),
            use_container_width=True,
        )

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(
                plot_pred_vs_actual(metrics["y_test"], metrics["pred_graph"], "GraphSAGE: Predicted vs. Actual"),
                use_container_width=True,
            )
        with col6:
            st.plotly_chart(
                plot_error_histogram(metrics["y_test"], metrics["pred_graph"], "GraphSAGE Prediction Error"),
                use_container_width=True,
            )
