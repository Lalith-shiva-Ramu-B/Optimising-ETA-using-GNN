import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from main import (corridor_setup,critical_hub_find,breached_find,time_bucket,GraphSAGE,
                  build_graph,graph_data,within_15_pct_accuracy,predict_dynamic_eta,RouteDecisionFramework)
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@400;600;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background-color: #0E1117;
}
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid rgba(255,255,255,0.1);
}
.stMetric {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    padding: 1rem;
}
div[role="radiogroup"] > label:hover {
    color: #4FC3F7;
    transition: color 0.2s ease;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 0;
    color: #8B949E;
}
.stTabs [aria-selected="true"] {
    color: #4FC3F7 !important;
    border-bottom: 2px solid #4FC3F7 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4FC3F7, #00BFA5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(79, 195, 247, 0.3) !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
}
[data-testid="stDataFrame"] th {
    background-color: rgba(79,195,247,0.1) !important;
    color: #E6EDF3 !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 1rem;
}
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0E1117; 
}
::-webkit-scrollbar-thumb {
    background: #30363D; 
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4FC3F7; 
}
[data-testid="stForm"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
}
.interactive-element:hover {
    transform: translateY(-2px);
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem 0 0.5rem 0;">
    <h1 style="font-family: 'Outfit', sans-serif; font-size: 2.4rem; 
        background: linear-gradient(135deg, #4FC3F7, #00BFA5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;">🚚 ETA Analytics & Routing Dashboard</h1>
    <p style="color: #8B949E; font-size: 1rem; font-family: 'Inter', sans-serif;">
        Monitor network performance · Identify bottlenecks · Optimize logistics routing</p>
</div>
""", unsafe_allow_html=True)
#st.divider()
# @st.cache_data
# def load_data(path):
#     data= pd.read_csv(path)
#     return data

@st.cache_data(show_spinner="Loading delivery dataset...")
def load_data():
    """Auto-load dataset from the Dataset folder."""
    data_path = Path(__file__).resolve().parent.parent / "Dataset" / "delivery_data.csv"
    if not data_path.exists():
        st.error(f"Dataset not found at: {data_path}")
        st.stop()
    return pd.read_csv(data_path)
df = load_data()

#upload_file=st.sidebar.file_uploader("Choose the file")
def section_header(title, subtitle=""):
    sub = f'<p style="color: #8B949E; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <h2 style="font-family: 'Outfit', sans-serif; color: #E6EDF3; margin-bottom: 0.2rem;">{title}</h2>
        {sub}
        <div style="height: 3px; width: 60px; background: linear-gradient(90deg, #4FC3F7, #00BFA5); border-radius: 2px; margin-top: 0.5rem;"></div>
    </div>
    """, unsafe_allow_html=True)
def render_kpi_card(icon, label, value, suffix=""):
    return f"""
    <div style="background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
                padding: 1.2rem; text-align: center; transition: transform 0.2s ease;">
        <div style="font-size: 1.8rem; margin-bottom: 0.3rem;">{icon}</div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 1.8rem; font-weight: 700;
                    background: linear-gradient(135deg, #4FC3F7, #00BFA5);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {value:,.0f}{suffix}</div>
        <div style="color: #8B949E; font-size: 0.85rem; margin-top: 0.3rem;
                    font-family: 'Inter', sans-serif;">{label}</div>
    </div>
    """

with st.spinner("Processing data..."):
    #df=load_data(upload_file)
    df = df[df['is_cutoff'] == True].copy()
    df.drop(columns=["route_schedule_uuid","cutoff_timestamp","trip_creation_time"],inplace=True)
    df['od_start_time'] = pd.to_datetime( df['od_start_time'], format='%d-%m-%Y %H:%M',errors='coerce')
    df['hour'] = df['od_start_time'].dt.hour 
    df['time_of_day'] =df['hour'].apply(time_bucket)
    ftl_df = df[df["route_type"] == "FTL"].copy()
    cart_df = df[df["route_type"] == "Carting"].copy()
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


# --- add these here, all in one place ---
@st.cache_data(show_spinner="Building corridor graph...")
def cached_corridor_setup(df: pd.DataFrame) -> pd.DataFrame:
    print(corridor_setup(df))
    return corridor_setup(df)

@st.cache_data(show_spinner="Finding breached corridors...")
def cached_breached_find(corridor_df: pd.DataFrame) -> pd.DataFrame:
    return breached_find(corridor_df)

@st.cache_data(show_spinner="Computing hub centrality metrics...")
def cached_critical_hub_find(corridor_df: pd.DataFrame):
    return critical_hub_find(corridor_df)

@st.cache_data(show_spinner="Building network graph...")
def cached_build_graph(corridor_df: pd.DataFrame):
    return build_graph(corridor_df)

@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH_f = BASE_DIR  / "ftl_model.joblib"
    MODEL_PATH_c = BASE_DIR  / "cart_model.joblib"
    return joblib.load(MODEL_PATH_f),joblib.load(MODEL_PATH_c)
ftl_model, cart_model = load_model()
corridor_df=cached_corridor_setup(df)
breached_df=cached_breached_find(corridor_df)
critical_hubs, hub_metrics_df=cached_critical_hub_find(corridor_df)

if page == "Executive Dashboard":
        st.subheader("Key Performance Indicators")
        column_1, column_2, column_3, column_4, column_5 =st.columns(5)
        with column_1:
             st.markdown(render_kpi_card("📦", "Total Trips", len(df)), unsafe_allow_html=True)
        with column_2:
               st.markdown(render_kpi_card("🏢", "Total Hubs", df["source_center"].nunique()), unsafe_allow_html=True)
        with column_3:
               st.markdown(render_kpi_card("🛣️", "Total Corridors", len(corridor_df)), unsafe_allow_html=True)
        with column_4:
                st.markdown(render_kpi_card("⚠️", "Breached Corridors", len(breached_df)), unsafe_allow_html=True)
        with column_5:
               st.markdown(render_kpi_card("⏱️", "Avg Delay Ratio", float(np.mean(corridor_df["median_delay_ratio"])), suffix="x"), unsafe_allow_html=True)

elif page == "Network Analysis":
    st.subheader("Interactive Corridor Network")
    st.caption("Visual representation of routing paths and bottlenecks.")
    graph = cached_build_graph(corridor_df)
    breached_edges = set(zip(breached_df["source_center"], breached_df["destination_center"]))
    fig = plot_network(graph, breached_edges=breached_edges, bottleneck_nodes=critical_hubs)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Hub Metrics")
    st.caption(f"Identified {len(critical_hubs)} structurally critical (bottleneck) hubs.")
    st.dataframe(hub_metrics_df.sort_values(by="Betweenness", ascending=False), use_container_width=True)

elif page == "Corridor Analysis":
    tab1, tab2, tab3 = st.tabs(["🚦 Top Delays & Breaches", "📊 Distribution Charts", "🔍 Search Corridors"])
    
    with tab1:
          colA, colB = st.columns(2)
          with colA:
            st.subheader("Top Delayed Corridors")
            top_delayed = corridor_df.sort_values(by="median_delay_ratio", ascending=False).head(15)
            st.dataframe(top_delayed, use_container_width=True,)
          with colB:
                st.subheader("Top Breached Corridors")
                st.dataframe(breached_df.head(15), use_container_width=True)
    with tab2:
            col1,col2,col3 = st.columns(3)
            with col1:
                st.plotly_chart(plot_delay_ratio_distribution(corridor_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_route_type_comparison(corridor_df), use_container_width=True)
            with col3:
                st.plotly_chart(plot_route_type_breaches(breached_df), use_container_width=True)


            st.plotly_chart(plot_time_of_day_breaches(breached_df), use_container_width=True)

            st.plotly_chart(plot_time_of_day_comparison(corridor_df), use_container_width=True)
    with tab3:
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
        hub_metrics_df.sort_values(by="SLA_Breach_Contribution", ascending=False).head(20),
        use_container_width=True,
    )

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "Betweenness"), use_container_width=True)
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "In_Degree"), use_container_width=True)
    with metric_cols[1]:
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "Out_Degree"), use_container_width=True)
        st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "SLA_Breach_Contribution"), use_container_width=True)

    st.plotly_chart(plot_bottleneck_bar(hub_metrics_df, "Clustering"), use_container_width=True)

elif page == "ML Model":
        
        st.subheader("Baseline (Random Forest) vs. GraphSAGE")
        with st.spinner("Generating predictions and calculating metrics..."):
                test_src_f, test_dst_f, tab_test_f,X_train_base_f, y_train_f,X_test_base_f,y_test_f,graph_f,node_mapping_f,scaler_f =graph_data(ftl_df)
                test_src_c, test_dst_c, tab_test_c,X_train_base_c, y_train_c,X_test_base_c,y_test_c,graph_c,node_mapping_c,scaler_c =graph_data(cart_df)
                model_f,model_c= load_model()
                with torch.no_grad():
                    pred_graph_f = model_f(graph_f.x, graph_f.edge_index, test_src_f, test_dst_f, tab_test_f).cpu().numpy()
                    pred_graph_c = model_c(graph_c.x, graph_c.edge_index, test_src_c, test_dst_c, tab_test_c).cpu().numpy()
                rf_base_c = RandomForestRegressor(n_estimators=100,random_state=42, n_jobs=-1)
                rf_base_c.fit(X_train_base_c, y_train_c)
                y_pred_base_c = rf_base_c.predict(X_test_base_c)

                rf_base_f = RandomForestRegressor(n_estimators=100,random_state=42, n_jobs=-1)
                rf_base_f.fit(X_train_base_f, y_train_f)
                y_pred_base_f = rf_base_f.predict(X_test_base_f)

                cart_results = cart_df[cart_df["data"] == "test"].copy()
                cart_results["graph_prediction"] = pred_graph_c
                #cart_results["rf_prediction"] = y_pred_base_c

                ftl_results = ftl_df[ftl_df["data"] == "test"].copy()
                ftl_results["graph_prediction"] = pred_graph_f
                #ftl_results["rf_prediction"] = y_pred_base_f

                base_mae = mean_absolute_error(y_test_c, y_pred_base_c)
                graph_mae = mean_absolute_error(y_test_c, pred_graph_c)
                base_acc = within_15_pct_accuracy(y_test_c, y_pred_base_c)
                graph_acc = within_15_pct_accuracy(y_test_c, pred_graph_c)
        tab_carting, tab_ftl = st.tabs(["🛒 Carting Routes", "🚚 FTL Routes"])
        with tab_carting:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Baseline MAE ", f"{base_mae:.2f}")
                col2.metric("GraphSAGE MAE ", f"{graph_mae:.2f}")
                col3.metric("Baseline within-15% accuracy", f"{base_acc:.1f}%")
                col4.metric("GraphSAGE within-15% accuracy", f"{graph_acc:.1f}%")
                
                st.plotly_chart(
                            plot_model_comparison_bar(
                                base_mae, graph_mae, base_acc,graph_acc,"Baseline vs. GraphSAGE Carting" ),
                            use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                            st.plotly_chart(
                                plot_pred_vs_actual(y_test_c, pred_graph_c, "GraphSAGE: Predicted vs. Actual for Carting Routes"),
                                use_container_width=True,
                            )
                with col2:
                            st.plotly_chart(
                                plot_error_histogram(y_test_c, pred_graph_c, "GraphSAGE Prediction Error for Carting Routes"),
                                use_container_width=True,
                            )
                base_mae_f = mean_absolute_error(y_test_f, y_pred_base_f)
                graph_mae_f = mean_absolute_error(y_test_f, pred_graph_f)
                base_acc_f = within_15_pct_accuracy(y_test_f, y_pred_base_f)
                graph_acc_f = within_15_pct_accuracy(y_test_f, pred_graph_f)
        with tab_ftl:
                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Baseline MAE ", f"{base_mae_f:.2f}")
                col6.metric("GraphSAGE MAE ", f"{graph_mae_f:.2f}")
                col7.metric("Baseline within-15% accuracy", f"{base_acc_f:.1f}%")
                col8.metric("GraphSAGE within-15% accuracy", f"{graph_acc_f:.1f}%")
                
                st.plotly_chart(
                    plot_model_comparison_bar(
                        base_mae_f, graph_mae_f, base_acc_f,graph_acc_f,"Baseline vs. GraphSAGE FTL"),
                    use_container_width=True,)

                
                col7, col8 = st.columns(2)
                with col7:
                            st.plotly_chart(
                                plot_pred_vs_actual(y_test_f, pred_graph_f, "GraphSAGE: Predicted vs. Actual for FTL Routes"),
                                use_container_width=True,
                            )
                with col8:
                            st.plotly_chart(
                                plot_error_histogram(y_test_f, pred_graph_f, "GraphSAGE Prediction Error for FTL Routes"),
                                use_container_width=True,
                            )
elif page == "Decision Framework":
    test_src_f, test_dst_f, tab_test_f,X_train_base_f, y_train_f,X_test_base_f,y_test_f,graph_f,node_mapping_f,scaler_f =graph_data(ftl_df)
    test_src_c, test_dst_c, tab_test_c,X_train_base_c, y_train_c,X_test_base_c,y_test_c,graph_c,node_mapping_c,scaler_c =graph_data(cart_df)
    model_f, model_c= load_model()
    with torch.no_grad():
             pred_graph_f = model_f(graph_f.x, graph_f.edge_index, test_src_f, test_dst_f, tab_test_f).cpu().numpy()
             pred_graph_c = model_c(graph_c.x, graph_c.edge_index, test_src_c, test_dst_c, tab_test_c).cpu().numpy()


    cart_results = cart_df[cart_df["data"] == "test"].copy()
    cart_results["graph_prediction"] = pred_graph_c
    
    ftl_results = ftl_df[ftl_df["data"] == "test"].copy()
    ftl_results["graph_prediction"] = pred_graph_f
    
    st.subheader("FTL vs. Carting Route Decision")
    st.caption(
        "Enter a shipment's parameters to get a cost/risk-based route recommendation, "
        "backed by the trained ETA model and the network's structural risk scores."
    )
    _src_names = df[["source_center", "source_name"]].drop_duplicates().dropna()
    _dst_names = df[["destination_center", "destination_name"]].drop_duplicates().dropna()
    centre_name_map = {}
    for _, row in _src_names.iterrows():
        centre_name_map[row["source_center"]] = row["source_name"]
    for _, row in _dst_names.iterrows():
        if row["destination_center"] not in centre_name_map:
            centre_name_map[row["destination_center"]] = row["destination_name"]
    def fmt_centre(code):
        """Display as 'CentreName  [Code]' so the selectbox is searchable by name."""
        name = centre_name_map.get(code, "")
        return f"{name}  [{code}]" if name else code
    # ── Build route adjacency (source → set of valid destinations) ──────
    route_pairs = corridor_df[["source_center", "destination_center"]].drop_duplicates()
    #src_to_dsts = route_pairs.groupby("source_center")["destination_center"].apply(lambda x: sorted(set(x))).to_dict()

    src_to_dsts = {}
    for src, dst in zip(corridor_df["source_center"], corridor_df["destination_center"]):
        if src != dst:                         # exclude self-loops
            src_to_dsts.setdefault(src, set()).add(dst)
    # convert sets to sorted lists for selectbox
    src_to_dsts = {k: sorted(v, key=fmt_centre) for k, v in src_to_dsts.items()}
    all_sources = sorted(set(corridor_df["source_center"]))

    #facilities_s = sorted(set(corridor_df["source_center"]) )
    #facilities_d = sorted(set(corridor_df["destination_center"]))
    #route_types = list(corridor_df["route_type"].unique())
    time_labels = {0: "Night", 1: "Morning", 2: "Afternoon", 3: "Evening"}
    
    c1, c2 = st.columns(2)
    with c1:
            source_center = st.selectbox( "Source hub",all_sources, format_func=fmt_centre,accept_new_options=False)
                                         #facilities_s,accept_new_options=False)

            source_name = df.loc[df["source_center"] == source_center,"source_name"].dropna().unique()

            # if len(source_name) > 0:
            #       st.write(source_name[0])
            # else:
            #    st.warning("Source name not found.")

            valid_destinations = src_to_dsts.get(source_center, [])
            if not valid_destinations:
               st.warning("No routes found from this source centre. Showing all destinations.")
            #valid_destinations = sorted(set(corridor_df["destination_center"]))
            valid_destinations = src_to_dsts.get(source_center, [])
            destination_center = st.selectbox("Destination hub", valid_destinations,
            format_func=fmt_centre,accept_new_options=False)
                                              #facilities_d, accept_new_options=False)
            destination_name = df.loc[ df["destination_center"] == destination_center,"destination_name"].dropna().unique()

            # if len(destination_name) > 0:
            #    st.write(destination_name[0])
            # else:
            #    st.warning("Destination name not found.")

            route_count = len(corridor_df[
            (corridor_df["source_center"] == source_center) &
            (corridor_df["destination_center"] == destination_center)
        ])
        

            if source_center == destination_center:
                  st.error("Source and destination are the same — pick two different hubs.")

            time_of_day = st.selectbox("Time of day", list(time_labels.keys()), format_func=lambda k: time_labels[k])
    with st.form("decision_form"):
        with c2:
            distance_km = corridor_df[(corridor_df["source_center"]==source_center) & (corridor_df["destination_center"]==destination_center)]["actual_distance_to_destination"].median()
            if pd.isna(distance_km):
                 st.error("Route is doesn't exist")
            volume = st.number_input("Shipment volume (packages)", min_value=1.0, value=200.0, step=10.0)
            sla_deadline_hours = st.number_input("SLA deadline (hours)", min_value=0.5, value=10.0, step=0.5)
        submitted = st.form_submit_button("Get Recommendation")

    risk_lookup = dict(zip(hub_metrics_df["Facility"],hub_metrics_df["SLA_Breach_Contribution"]))
    if submitted:
        #print("distnace travelled: ",distance_km)
        ftl_median = ftl_results[(ftl_results["source_center"]==source_center) & (ftl_results["destination_center"]==destination_center)]["graph_prediction"].median()
        
        if pd.isna(ftl_median):
            pred_etl_ftl = predict_dynamic_eta(model_f, graph_f, ftl_df, source_center, destination_center, node_mapping_f, scaler_f)
        else:
            pred_etl_ftl = ftl_median
            
        # --- 2. Independently evaluate Carting ---
        cart_median = cart_results[(cart_results["source_center"]==source_center) & (cart_results["destination_center"]==destination_center)]["graph_prediction"].median()
        
        if pd.isna(cart_median):
            pred_etl_carting = predict_dynamic_eta(model_c, graph_c, cart_df, source_center, destination_center, node_mapping_c, scaler_c)
        else:
            pred_etl_carting = cart_median
        framework = RouteDecisionFramework(hub_risk_lookup=risk_lookup) 
        decision = framework.evaluate_tradeoff(
        distance_km=distance_km,
        current_volume=volume,
        pred_eta_ftl=pred_etl_ftl,     # Using median prediction for FTL
        pred_eta_carting=pred_etl_carting,  # Using median prediction for Carting
        sla_deadline=sla_deadline_hours,
        source_center=source_center , # High network risk
        destination_center=destination_center,
        time_of_day=time_of_day
)
        # for key, value in decision.items():
        #   st.write(f"{key}: {value}")
        if decision['Recommendation'] == 'FTL':
            rec_color, rec_icon = '#4FC3F7', '🚛'
        else:
            rec_color, rec_icon = '#00BFA5', '🛒'
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
                    border: 1px solid {rec_color}40; border-radius: 16px; padding: 2rem; margin: 1rem 0;">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <span style="font-size: 3rem;">{rec_icon}</span>
                <h2 style="font-family: 'Outfit', sans-serif; color: {rec_color}; margin: 0.5rem 0 0.2rem 0;">
                    Recommended: {decision['Recommendation']}</h2>
                <p style="color: #8B949E; font-style: italic;">{decision['Reasoning']}</p>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="background: rgba(79,195,247,0.1); border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="color: #8B949E; font-size: 0.8rem;">FTL ETA</div>
                    <div style="color: #4FC3F7; font-size: 1.4rem; font-weight: 700;">{decision['predicted_ETA_FTL']} hrs</div>
                    <div style="color: #8B949E; font-size: 0.8rem; margin-top: 0.3rem;">₹{decision['Est_Cost_per_Unit_FTL']}/pkg</div>
                </div>
                <div style="background: rgba(0,191,165,0.1); border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="color: #8B949E; font-size: 0.8rem;">Carting ETA</div>
                    <div style="color: #00BFA5; font-size: 1.4rem; font-weight: 700;">{decision['predicted_ETA_Carting']} hrs</div>
                    <div style="color: #8B949E; font-size: 0.8rem; margin-top: 0.3rem;">₹{decision['Est_Cost_per_Unit_Carting']}/pkg</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <span style="color: #FFB74D; font-weight: 600;">Utility Score Difference: ₹{decision['Utility_Score_Difference']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    if submitted is False:
        st.info("Fill in the shipment parameters and click 'Get Recommendation' to see the route decision.")