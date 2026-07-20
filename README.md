<html lang ="en">

# Optimizing Delivery Estimated Time of Arrival (ETAs) with Graph-Based Network Intelligence

Developed for IIT guwahati Summer Project

# Table of Content
1. Problem Statement
2. Features
3. Dataset
4. Methodology
5. Model Architecture
6. Repository Structure
7. Installation
8. Usage
9. Results
10. Future Improvements

# Problem Statement

<p><b><h2>Background</h2></b>
Delhivery is India's largest fully-integrated logistics provider, operating a vast network of facilities, intercity routes, and last-mile delivery across every major state. At the core of its operations is a hub-and spoke model: shipments travel from a source facility through one or more intermediate hubs beforereaching the destination, each hop is a segment of a multi-leg journey.
To estimate delivery times, Delhivery uses OSRM, a standard routing engine that assumes clean traffic and shortest paths. But real-world logistics is far messier: congestion, facility dwell time, seasonal volume spikes, and route-type constraints all cause actual delivery times to deviate significantly from predictions.
<b><h2>The Challenge</h2></b>
The strategic question: Delhivery's OSRM system underestimates actual delivery time on a significant fraction of routes. Can a graph-based model, one that treats the logistics network as a connected graph of facilities and corridors, not a collection of independent point-to-point estimates, produce more accurate ETAs and identify which corridors and hubs are systematically causing delays? When ETA is wrong, SLAs are missed and customers are unhappy Downstream capacity planning breaks down across the network
There is currently no systematic way to identify which hubs and corridors are the biggest contributors to delays Route-type decisions (FTL vs Carting) are made without accounting for graph position or structural risk of a facility
</p>

# Features
1. Hub Risk Analysis
2. Corridor Risk Analysis
3. Logistics Network modelling as Graph
4. Graph-based ETA prediction using Graphsage
5. Route Decision Framework
6. Interactive Streamlit DashBoard 

# Methodolgy

Raw Shipement Data\
Hub Risk Analysis\
Graph Construction\
Node Feature Generation\
Graphsage Training\
ETA Prediction\
Risk Scoring\
DashBoard\

# Repository structure

This respository contains 3 folders 
<ul>
<li>Dataset  </li>
<li>python_files</li>
<li>Strategy Memo</li>
</ul> 

# Dataset 
<p> It has the CSV file used for network Analysis and ETA prediction. It has both train and test data combined </p>

# Python_Files
<p> It has 3 Python files </p>
<ul>
<li>main.py : Contains all preprocessing,Graph Constructions, Graphsage model and Route Decision Framework</li>
<li>plot.py : Contains all type of custom functions for plotting the graphs used for making dashboard</li>
<li>streamlit.py : Contains all the streamlit app code for dashboard and prediction of Route Decision Framework</li> 
</ul>

# Strategy Memo 
Contains the microsoft doc file of strategy memo

# Installation 
clone the repository 

git clone
cd ETA_ML_PROJECT
streamlit run streamlit_app.py

# Usage
open the streamlit_app in python_files folder then you can open the streamlit application locally 

#   Results
Succesfully Identified high risked hubs\
Improvised ETA prediction compared to OSRM baseline\
Corridor level risk analysis\
Interactive business dashboard

## Future improvements 

1. deploy on Streamlit Cloud
2. Hyperparameter tuning

## Author
Boddu Lalith Shiva Ramu\
B.Tech Mechanical + MBA dual Degree\
Supply Chain Analytics | Machine Learning | Graphical Neural Networks 

