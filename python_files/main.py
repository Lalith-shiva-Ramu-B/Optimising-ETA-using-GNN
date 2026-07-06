import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import SAGEConv
import networkx as nx
from torch_geometric.data import Data
import seaborn as sns
import matplotlib.pyplot as plt
import random 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from streamlit_app import df

SEED = 42
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # local import: torch is an optional/heavy dependency

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def corridor_setup(df):
    df['is_breach']= (df['actual_time'] > (df['osrm_time'] * 1.20)).astype(int)
    df['breach_time']=( df['actual_time']-( df['osrm_time']))
    df['delay_ratio'] = df['actual_time'] / df['osrm_time']
    df['od_start_time'] = pd.to_datetime( df['od_start_time'], format='%d-%m-%Y %H:%M',errors='coerce')
    df['hour'] =    df['od_start_time'].dt.hour  
    df=df[df['is_cutoff']== True] # droped is_cutoff = false since no pattern followed 
    def time_bucket(hour):
        if (0 <= hour < 6) or (22<=hour<24): 
            return 0      # Night
        elif 6 <= hour < 12: 
            return 1   # Morning
        elif 12 <= hour < 18: 
            return 2  # Afternoon
        else: 
            return 3   # Evening

    df['time_of_day'] =df['hour'].apply(time_bucket)

# Identify Chronically Delayed Corridors (Median delay > 20%)
    corridor_df = df.groupby(['source_center', 'destination_center','route_type','time_of_day']).agg(
    median_actual=('segment_actual_time', 'median'),
    median_osrm=('segment_osrm_time', 'median'),
    total_trips=('trip_uuid', 'count'), 
    breaches=('is_breach','sum'),
    osrm_distance=('segment_osrm_distance', 'max'),
    actual_distance_to_destination=('actual_distance_to_destination', 'max'),
    breach_time=('breach_time','sum')
).reset_index()
    corridor_df['median_delay_ratio']=corridor_df['median_actual']/corridor_df['median_osrm']
    print("total cooridors :", len(corridor_df))
    print("no of breached corridors",len(corridor_df[corridor_df['breaches']>0])) 
    return corridor_df

def critical_hub_find(corridor_df):
    G_nx = nx.DiGraph()
# Add edges with the median delay ratio acting as the mathematical weight
    for _, row in corridor_df.iterrows():
            G_nx.add_edge(row['source_center'], row['destination_center'], weight=row['median_delay_ratio'])
    dict_in_degree = dict(G_nx.in_degree())
    dict_out_degree = dict(G_nx.out_degree())
    dict_betweenness = nx.betweenness_centrality(G_nx, weight='weight',normalized=True)
    dict_clustering = nx.clustering(G_nx.to_undirected())

    hub_breach_counts = {node: 0 for node in G_nx.nodes()}  

    for _, row in corridor_df.iterrows():
             b = float(row["breaches"])
             hub_breach_counts[row["source_center"]] = hub_breach_counts.get(row["source_center"], 0.0) + b
             hub_breach_counts[row["destination_center"]] = hub_breach_counts.get(row["destination_center"], 0.0) + b

    total_network_breaches = corridor_df['breaches'].sum()
    hub_metrics_df = pd.DataFrame({
    'Facility': list(G_nx.nodes()),
    'In_Degree': [dict_in_degree[n] for n in G_nx.nodes()],
    'Out_Degree': [dict_out_degree[n] for n in G_nx.nodes()],
    'Betweenness': [dict_betweenness[n] for n in G_nx.nodes()],
    'Clustering': [dict_clustering[n] for n in G_nx.nodes()],
    'Total_Breaches': [hub_breach_counts[n] for n in G_nx.nodes()]
})
    if total_network_breaches > 0:
        hub_metrics_df['SLA_Breach_Contribution_%'] = (hub_metrics_df['Total_Breaches'] / total_network_breaches) * 100
    else:
        hub_metrics_df['SLA_Breach_Contribution_%'] = 0

    top_in_degree= hub_metrics_df[hub_metrics_df['In_Degree']>=10].sort_values(by='In_Degree', ascending=False).reset_index(drop=True)
    top_in_degree=top_in_degree[['Facility', 'In_Degree']]
    top_indegree_set=set(top_in_degree['Facility'])

    top_out_degree= hub_metrics_df[hub_metrics_df['Out_Degree']>=10].sort_values(by='Out_Degree', ascending=False).reset_index(drop=True)
    top_out_degree=top_out_degree[['Facility', 'Out_Degree']]
    top_outdegree_set=set(top_out_degree['Facility'])

    top_betweeness= hub_metrics_df[hub_metrics_df['Betweenness']>=0.03 ].sort_values(by='Betweenness', ascending=False).reset_index(drop=True)
    top_betweeness=top_betweeness[['Facility', 'Betweenness']]
    betweeness_set=set(top_betweeness['Facility'])

    top_clustering= hub_metrics_df[hub_metrics_df['Clustering']<=0.2 ].sort_values(by='Clustering', ascending=False).reset_index(drop=True)
    top_clustering=top_clustering[['Facility', 'Clustering']]
    clustering_set=set(top_clustering['Facility'])

    final_degree_set=top_indegree_set.intersection(top_outdegree_set)
    final_set1=final_degree_set.intersection(betweeness_set)
    critical_hubs=final_set1.intersection(clustering_set)
    print("final critical hubs : ", critical_hubs)
    top_breaches = hub_metrics_df[hub_metrics_df['Total_Breaches'] >= 20].sort_values(by='Total_Breaches', ascending=False).reset_index(drop=True)
    keys=list(critical_hubs)
    return critical_hubs,hub_metrics_df

def breached_find(corridor_df):
    corridor_df['sla_breach_rate']=corridor_df['breaches']/corridor_df['total_trips']
    breached_coordiors= corridor_df[corridor_df['breaches']>0].sort_values(by='sla_breach_rate', ascending=False).reset_index(drop=True)
    print("breached corridors : ",len(breached_coordiors))
    #print(breached_coordiors[['source_center','destination_center','route_type','total_trips','breaches','breach_time','time_of_day','sla_breach_rate','median_delay_ratio']].head(5))
    return breached_coordiors

corridor_df=corridor_setup(df)
breached_df=breached_find(corridor_df)
critical_hubs, hub_metrics_df=critical_hub_find(corridor_df)
def build_graph(corridor_df: pd.DataFrame, weight_col: str = "median_delay_ratio"):
    graph = nx.DiGraph()
    for _, row in corridor_df.iterrows():
        graph.add_edge(row["source_center"], row["destination_center"], weight=row[weight_col])
    return graph

def encode_route_types(df: pd.DataFrame, corridor_df: pd.DataFrame):
    df = df.copy()
    unique_routes = corridor_df["route_type"].unique()
    route_mapping = {route: idx for idx, route in enumerate(unique_routes)}
    print("route mapping:", route_mapping)
    df["route_type_encoded"] = df["route_type"].map(route_mapping).fillna(0).astype(int)
    return df, route_mapping
df,route_mapping=encode_route_types(df,corridor_df)
BASE_FEATURES = ["segment_osrm_time", "segment_osrm_distance", "time_of_day", "route_type_encoded"]
def prepare_graphsage_dataset(
    df: pd.DataFrame,
    corridor_df: pd.DataFrame):
     set_seed(SEED)
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     test_df = df[df["data"] == "test"].copy()
     train_df = df[df["data"] == "training"].copy()

     train_df[BASE_FEATURES] = train_df[BASE_FEATURES].fillna(0)
     test_df[BASE_FEATURES] = test_df[BASE_FEATURES].fillna(0)

     x_train_base_raw = train_df[BASE_FEATURES].values
     x_test_base_raw = test_df[BASE_FEATURES].values
     scaler = StandardScaler()
     X_train_base = scaler.fit_transform(x_train_base_raw)
     X_test_base = scaler.transform(x_test_base_raw)

     y_train = train_df["segment_actual_time"].values.astype(np.float32)
     y_test = test_df["segment_actual_time"].values.astype(np.float32)


# 4. PREPARE SEPARATE TRAIN & TEST PARAMETERS
     all_nodes = pd.concat([
    train_df["source_center"], train_df["destination_center"],
    test_df["source_center"], test_df["destination_center"]
]).dropna().unique()

     node_mapping = {node: i for i, node in enumerate(all_nodes)}
     num_nodes = len(all_nodes)

     for frame in [train_df, test_df]:
         frame["src_idx"] = frame["source_center"].map(node_mapping)
         frame["dst_idx"] = frame["destination_center"].map(node_mapping)

     train_df = train_df.dropna(subset=["src_idx", "dst_idx"]).copy()
     test_df = test_df.dropna(subset=["src_idx", "dst_idx"]).copy()

     train_df["src_idx"] = train_df["src_idx"].astype(int)
     train_df["dst_idx"] = train_df["dst_idx"].astype(int)
     test_df["src_idx"] = test_df["src_idx"].astype(int)
     test_df["dst_idx"] = test_df["dst_idx"].astype(int)

# Node features built only from training graph
     src_counts = np.bincount(train_df["src_idx"].values, minlength=num_nodes)
     dst_counts = np.bincount(train_df["dst_idx"].values, minlength=num_nodes)
     tot_counts = src_counts + dst_counts

     node_features = np.stack([
    np.log1p(src_counts),
    np.log1p(dst_counts),
    np.log1p(tot_counts),
], axis=1).astype(np.float32)

     x = torch.tensor(node_features, dtype=torch.float)

     edge_index = torch.tensor(
    np.vstack([train_df["src_idx"].values, train_df["dst_idx"].values]),
    dtype=torch.long
)

# Make message passing stronger by adding reverse edges
     edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

     graph_data = Data(x=x, edge_index=edge_index)
     graph_data = graph_data.to(device)
     train_df = train_df.sort_values(by=["src_idx", "dst_idx"]).reset_index(drop=True)
     test_df = test_df.sort_values(by=["src_idx", "dst_idx"]).reset_index(drop=True)

     train_src = torch.tensor( train_df["src_idx"].values, dtype=torch.long, device=device)
     train_dst = torch.tensor(train_df["dst_idx"].values, dtype=torch.long, device=device)
     test_src = torch.tensor(test_df["src_idx"].values, dtype=torch.long, device=device)
     test_dst = torch.tensor(test_df["dst_idx"].values, dtype=torch.long, device=device)

     tab_train = torch.tensor(X_train_base, dtype=torch.float32, device=device)
     tab_test = torch.tensor(X_test_base, dtype=torch.float32, device=device)
     y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
     y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

     return graph_data,train_src,train_dst,y_train_t,y_test_t ,test_src,test_dst, tab_train,tab_test, y_train,y_test,scaler,route_mapping,train_df,test_df,X_train_base,X_test_base
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, tabular_dim, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) 
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, src_idx, dst_idx, tabular_x):
        z = self.encode(x, edge_index)
        edge_z = torch.cat([z[src_idx], z[dst_idx], tabular_x], dim=1)
        out = self.mlp(edge_z).squeeze(-1)
        return out
    
def train_model(graph_data,train_src,train_dst ,y_train_t,y_test_t,test_src,test_dst, tab_train,tab_test, y_train,y_test,scaler,route_mapping,train_df,test_df,X_train_base,X_test_base
):
    model = GraphSAGE(
    in_channels=graph_data.x.shape[1],
    hidden_channels=32,
    tabular_dim=tab_train.shape[1],
    dropout=0.2
)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    print("Graph Sage model Training started")
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        pred = model(graph_data.x, graph_data.edge_index, train_src, train_dst, tab_train)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
    

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_graph = model(graph_data.x, graph_data.edge_index, test_src, test_dst, tab_test).cpu().numpy()

    rf_base = RandomForestRegressor(n_estimators=100,random_state=SEED, n_jobs=-1)
    rf_base.fit(X_train_base, y_train)
    y_pred_base = rf_base.predict(X_test_base)

    return y_pred_base,pred_graph

def within_15_pct_accuracy(y_true, y_pred):
    # Prevent division by zero
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    error_ratio = np.abs(y_true - y_pred) / y_true_safe
    return np.mean(error_ratio <= 0.15) * 100

    base_mae = mean_absolute_error(y_test, y_pred_base)
    graph_mae = mean_absolute_error(y_test, pred_graph)

    base_acc = within_15_pct_accuracy(y_test, y_pred_base)
    graph_acc = within_15_pct_accuracy(y_test, pred_graph)