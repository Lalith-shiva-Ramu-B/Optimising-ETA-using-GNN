import pandas as pd
import numpy as np
import re
from pathlib import Path
import pgeocode
import folium
from folium.plugins import MarkerCluster
from main import corridor_setup,time_bucket
#@st.cache_data(show_spinner="Loading delivery dataset...")
def load_data():
    """Auto-load dataset from the Dataset folder."""
    data_path = Path(__file__).resolve().parent.parent / "Dataset" / "delivery_data.csv"
    return pd.read_csv(data_path)
MANUAL_COORDS = {
    "IND000000AAL": ("Pune", "Maharashtra", 18.5204, 73.8567),
    "IND000000AAS": ("Bongaigaon", "Assam", 26.4769, 90.5584),
    "IND000000AAZ": ("Buldhana", "Maharashtra", 20.5293, 76.1844),
    "IND000000ABA": ("Kochi", "Kerala", 10.0159, 76.2841),
    "IND000000ABD": ("Palakkad", "Kerala", 10.7867, 76.6548),
    "IND000000ABG": ("Hyderabad", "Telangana", 17.4400, 78.3489),
    "IND000000ACA": ("Ludhiana", "Punjab", 30.9010, 75.8573),
    "IND000000ACB": ("Gurgaon", "Haryana", 28.4595, 77.0266),
    "IND000000ACK": ("Hyderabad", "Telangana", 17.3616, 78.4747),
    "IND000000ACN": ("Hyderabad", "Telangana", 17.4065, 78.4772),
    "IND000000ACO": ("Kasaragod", "Kerala", 12.4996, 74.9869),
    "IND000000ACS": ("Bengaluru", "Karnataka", 12.9141, 77.6411),
    "IND000000ACT": ("Delhi", "Delhi", 28.6508, 77.2167),
    "IND000000ADM": ("Kochi", "Kerala", 9.9816, 76.2999),
    "IND000000ADV": ("Chennai", "Tamil Nadu", 13.1478, 80.2307),
    "IND000000AEL": ("Delhi", "Delhi", 28.6341, 77.2143),
    "IND000000AEM": ("Gurgaon", "Haryana", 28.3670, 76.9580),
    "IND000000AET": ("Barpeta", "Assam", 26.3177, 91.0064),
    "IND000000AFF": ("Darjeeling", "West Bengal", 27.0360, 88.2627),
    "IND000000AFG": ("Tirur", "Kerala", 10.9146, 75.9225),
    "IND000000AFJ": ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
    "IND000000AFR": ("Kochi", "Kerala", 9.9674, 76.2454),
    "IND000000AFS": ("Mumbai", "Maharashtra", 19.0990, 72.8740),
    "IND000000AFT": ("Navi Mumbai", "Maharashtra", 19.0440, 73.0190),
    "IND122050AAA":("Manesar","Haryana",28.3500,76.94),
    "IND142401AAA":("Khanna","Punjab",30.7114,76.2142),
    "IND205135AAA":("Shikohabad","Uttar Pradesh",27.1084,78.5846),
    "IND212501AAA":("Soraon", "Uttar Pradesh",25.6082,81.8496),
    "IND263002AAA": ("Nainital_Sookhtal_D", "Uttarakhand", 29.3919, 79.4542),
    "IND263401AAA": ("Kashipur_BajprDPP_D", "Uttarakhand", 29.2104, 78.9619),
    "IND275503AAA": ("Ghosi_Jamalpur_D", "Uttar Pradesh", 26.1118, 83.5414),
    "IND302014AAA": ("Jaipur_Hub", "Rajasthan", 26.8378, 75.7725),
    "IND302014AAB": ("Jaipur_Central_I_7", "Rajasthan", 26.8385, 75.7730),
    "IND302023AAA": ("Ambabadi_DC", "Rajasthan", 26.9406, 75.7820),
    "IND396232AAA": ("Silvassa_Samrvrni_D", "Dadra and Nagar Haveli", 20.2647, 73.0039),
    "IND400705AAA": ("Mumbai_Sanpada_CP", "Maharashtra", 19.0664, 73.0084),
    "IND401104AAA": ("Mumbai_MiraRd_IP", "Maharashtra", 19.2813, 72.8688),
    "IND401104AAB": ("Mumbai_MiraRoad_M", "Maharashtra", 19.2815, 72.8690),
    "IND410209AAA": ("Mumbai_Khandeshwar_Dc", "Maharashtra", 19.0143, 73.0945),
    "IND421802AAA": ("Khandala_Satara_D", "Maharashtra", 18.1256, 74.0135),
    "IND441603AAA": ("Chamorshi_Central_DPP_1", "Maharashtra", 19.9238, 79.8839),
    "IND462021AAA": ("Bhopal_Indrapri_DC", "Madhya Pradesh", 23.2505, 77.4641),
    "IND492007AAA": ("Raipur_Central_D_5", "Chhattisgarh", 21.2582, 81.6508),
    "IND509124AAA": ("JoguGadwal_ColctrOf_D", "Telangana", 16.2307, 77.8016),
    "IND632402AAA": ("Ranipet_MBTRd_DC", "Tamil Nadu", 12.9292, 79.3178),
    "IND686028AAB": ("Kottayam_Central_H_1", "Kerala", 9.5997, 76.5367),
    "IND713364AAB": ("Rupnarayanpur_Salanpur_D", "West Bengal", 23.8211, 86.8997),
    "IND811399AAA": ("Noida_Sector02_C", "Uttar Pradesh", 28.5835, 77.3168),
    "IND852118AAA": ("Forbesganj_JhumanCk_D", "Bihar", 26.2991, 87.2764)
}

# The single 5-digit PIN code centre
FIVE_DIGIT_FIX = {
    "IND68004AAA": "680004",  # Thrissur, Kerala
}
def extract_pin(centre_code):
    """Extract 6-digit PIN code from centre code like IND388121AAA."""
    if centre_code in FIVE_DIGIT_FIX:
        return FIVE_DIGIT_FIX[centre_code]
    m = re.match(r"IND(\d{6})", centre_code)
    if m and m.group(1) != "000000":
        return m.group(1)
    return None

def extract_city_state(centre_name):
    """Extract city and state from name like 'Pune_Tathawde_H (Maharashtra)'."""
    state = None
    sm = re.search(r"\(([^)]+)\)", centre_name)
    if sm:
        state = sm.group(1)
    if centre_name :
        city = centre_name.split("_")[0]  
    else:
        city = None
    return city, state


def geocode_all_centres(dataset_path):
    """Main geocoding pipeline."""
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    # Build centre_code -> centre_name mapping from both source and destination
    mapping = {}
    #if "source_name" in df.columns:
    for _, row in df[["source_center", "source_name"]].drop_duplicates().dropna().iterrows():
            mapping[row["source_center"]] = row["source_name"]
    #if "destination_name" in df.columns:
    for _, row in df[["destination_center", "destination_name"]].drop_duplicates().dropna().iterrows():
            if row["destination_center"] not in mapping:
                mapping[row["destination_center"]] = row["destination_name"]
    all_centres = sorted(mapping.keys())
    print(f"Total unique centres: {len(all_centres)}")
    # Initialize pgeocode for India
    nomi = pgeocode.Nominatim("in")
    results = []
    geocoded = 0
    manual = 0
    failed = 0
    for code in all_centres:
        name = mapping.get(code, "")
        city, state = extract_city_state(name)
        # 1. Check manual override first (IND000000 codes)
        if code in MANUAL_COORDS:
            mc = MANUAL_COORDS[code]
            results.append({
                "centre_code": code,
                "centre_name": name,
                "city": mc[0],
                "state": mc[1],
                "latitude": mc[2],
                "longitude": mc[3],
                "geocode_method": "manual",
                "needs_review": False,
            })
            manual += 1
            continue
        # 2. Try PIN code lookup via pgeocode
        pin = extract_pin(code)
        if pin:
            result = nomi.query_postal_code(pin)
            if not np.isnan(result.latitude) and not np.isnan(result.longitude):
                lat, lng = float(result.latitude), float(result.longitude)
                # Validate Indian bounds
                if 6.0 <= lat <= 37.0 and 68.0 <= lng <= 98.0:
                    results.append({
                        "centre_code": code,
                        "centre_name": name,
                        "city": city or (result.place_name if hasattr(result, "place_name") else ""),
                        "state": state or (result.state_name if hasattr(result, "state_name") else ""),
                        "latitude": round(lat, 6),
                        "longitude": round(lng, 6),
                        "geocode_method": "pgeocode_pin",
                        "needs_review": False,
                    })
                    geocoded += 1
                    continue
        # 3. Failed — flag for manual review
        results.append({
            "centre_code": code,
            "centre_name": name,
            "city": city or "",
            "state": state or "",
            "latitude": None,
            "longitude": None,
            "geocode_method": "FAILED",
            "needs_review": True,
        })
        failed += 1
    # Create output DataFrame
    out_df = pd.DataFrame(results)
    # Summary
    print(f"\n{'='*50}")
    print(f"GEOCODING RESULTS")
    print(f"{'='*50}")
    print(f"  PIN code (pgeocode) : {geocoded}")
    print(f"  Manual override     : {manual}")
    print(f"  FAILED              : {failed}")
    print(f"  Total               : {len(results)}")
    return out_df

coords_df= geocode_all_centres("C:/Users/lalit/Downloads/ETA_ML_Project/Dataset/delivery_data.csv")

PRIMARY = "#4FC3F7"
SECONDARY = "#00BFA5"
WARNING = "#FFB74D"
ERROR = "#EF5350"
ACCENT = "#AB47BC"
DARK_BG = "#1A1D23"

def lookup_df(df):
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["centre_code"]] = (
            row["latitude"],
            row["longitude"],
            row.get("centre_name", ""),
            row.get("city", ""),
            row.get("state", ""),
        )
    return lookup
     
def popup_html(code, name, city, state, lat, lng, extra_info=None):
    """Generate styled popup HTML for a centre marker."""
    extra = ""
    if extra_info:
        for k, v in extra_info.items():
            extra += f'<div style="font-size:11px;color:#aaa;">{k}: <b>{v}</b></div>'
    return f"""
    <div style="font-family:Inter,sans-serif;min-width:180px;">
        <div style="font-weight:700;font-size:13px;color:#E6EDF3;margin-bottom:4px;">{name}</div>
        <div style="font-size:11px;color:#8B949E;">{city}, {state}</div>
        <div style="font-size:10px;color:#666;margin-top:2px;">{code}</div>
        <div style="font-size:10px;color:#555;">({lat:.4f}, {lng:.4f})</div>
        {extra}
    </div>
    """     

def create_india_overview_map(
    coords_df,
    bottleneck_hubs,
    breached_corridors_df,
    height= 600,
):
    """
    Create a full-India overview map with all logistics centres.
    - All centres shown as clustered markers
    - Bottleneck hubs highlighted in orange
    - Breached corridors drawn as red lines
    """
    bottleneck_hubs = bottleneck_hubs or set()
    lookup = lookup(coords_df)
    m = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    # ── Marker cluster for all centres ──────────────────────────────────────
    cluster = MarkerCluster(
        name="Logistics Centres",
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 10},
    ).add_to(m)
    for _, row in coords_df.iterrows():
        code = row["centre_code"]
        lat, lng = row["latitude"], row["longitude"]
        name = row.get("centre_name", code)
        city = row.get("city", "")
        state = row.get("state", "")
        is_bottleneck = code in bottleneck_hubs
        color = WARNING if is_bottleneck else PRIMARY
        radius = 8 if is_bottleneck else 4
        popup_content = popup_html(
            code, name, city, state, lat, lng,
            {"Role": "⚠️ Bottleneck Hub"} if is_bottleneck else None,
        )
        folium.CircleMarker(
            location=[lat, lng],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7 if is_bottleneck else 0.5,
            weight=2 if is_bottleneck else 1,
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=name,
        ).add_to(cluster)
    # ── Breached corridor lines ─────────────────────────────────────────────
    if breached_corridors_df is not None and not breached_corridors_df.empty:
        breach_group = folium.FeatureGroup(name="Breached Corridors")
        drawn = 0
        for _, row in breached_corridors_df.iterrows():
            src = row.get("source_center")
            dst = row.get("destination_center")
            if src in lookup and dst in lookup:
                s_lat, s_lng = lookup[src][0], lookup[src][1]
                d_lat, d_lng = lookup[dst][0], lookup[dst][1]
                delay = row.get("median_delay_ratio", "")
                tooltip = f"{src} → {dst}"
                if delay:
                    tooltip += f" | Delay: {delay:.2f}x"
                folium.PolyLine(
                    locations=[[s_lat, s_lng], [d_lat, d_lng]],
                    color=ERROR,
                    weight=2,
                    opacity=0.5,
                    tooltip=tooltip,
                ).add_to(breach_group)
                drawn += 1
        breach_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m     

def create_route_map(
    coords_df,
    source_code,
    dest_code,
    source_name ,
    dest_name ,
    route_info,
    show_nearby,
    height = 500,
) :
    """
    Create a focused map showing source → destination route.
    Parameters
    ----------
    route_info : dict, optional
        Additional info to display (e.g., predicted ETA, cost, recommendation).
    show_nearby : bool
        If True, also show other centres near the route.
    """
    lookup = lookup_df(coords_df)
    if source_code not in lookup or dest_code not in lookup:
        return None
    s_lat, s_lng, s_name, s_city, s_state = lookup[source_code]
    d_lat, d_lng, d_name, d_city, d_state = lookup[dest_code]
    # Center and zoom to fit both points
    center_lat = (s_lat + d_lat) / 2
    center_lng = (s_lng + d_lng) / 2
    # Estimate zoom from distance
    lat_diff = abs(s_lat - d_lat)
    lng_diff = abs(s_lng - d_lng)
    max_diff = max(lat_diff, lng_diff)
    if max_diff < 0.5:
        zoom = 11
    elif max_diff < 2:
        zoom = 9
    elif max_diff < 5:
        zoom = 7
    elif max_diff < 10:
        zoom = 6
    else:
        zoom = 5
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    # ── Source marker (green) ───────────────────────────────────────────────
    src_extra = {"Type": "📍 Source"}
    if route_info:
        if "predicted_ETA_FTL" in route_info:
            src_extra["FTL ETA"] = f"{route_info['predicted_ETA_FTL']} hrs"
    folium.Marker(
        location=[s_lat, s_lng],
        popup=folium.Popup(
            popup_html(source_code, source_name or s_name, s_city, s_state, s_lat, s_lng, src_extra),
            max_width=280,
        ),
        tooltip=f"📍 Source: {source_name or s_name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    # ── Destination marker (red) ────────────────────────────────────────────
    dst_extra = {"Type": "🏁 Destination"}
    if route_info:
        if "predicted_ETA_Carting" in route_info:
            dst_extra["Carting ETA"] = f"{route_info['predicted_ETA_Carting']} hrs"
    folium.Marker(
        location=[d_lat, d_lng],
        popup=folium.Popup(
            popup_html(dest_code, dest_name or d_name, d_city, d_state, d_lat, d_lng, dst_extra),
            max_width=280,
        ),
        tooltip=f"🏁 Destination: {dest_name or d_name}",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(m)
    # ── Route line ──────────────────────────────────────────────────────────
    route_color = PRIMARY
    route_tooltip = f"{source_name or s_name} → {dest_name or d_name}"
    if route_info:
        rec = route_info.get("Recommendation", "")
        if rec == "FTL":
            route_color = PRIMARY
        elif rec == "Carting":
            route_color = SECONDARY
        route_tooltip += f" | Rec: {rec}"
    folium.PolyLine(
        locations=[[s_lat, s_lng], [d_lat, d_lng]],
        color=route_color,
        weight=4,
        opacity=0.8,
        tooltip=route_tooltip,
        dash_array="10",
    ).add_to(m)
    # ── Nearby centres ──────────────────────────────────────────────────────
    if show_nearby:
        nearby_group = folium.FeatureGroup(name="Nearby Centres")
        margin = max(lat_diff, lng_diff) * 0.3 + 0.5
        for _, row in coords_df.iterrows():
            code = row["centre_code"]
            if code in (source_code, dest_code):
                continue
            rlat, rlng = row["latitude"], row["longitude"]
            if (min(s_lat, d_lat) - margin <= rlat <= max(s_lat, d_lat) + margin and
                min(s_lng, d_lng) - margin <= rlng <= max(s_lng, d_lng) + margin):
                folium.CircleMarker(
                    location=[rlat, rlng],
                    radius=3,
                    color="#555",
                    fill=True,
                    fill_color="#555",
                    fill_opacity=0.4,
                    tooltip=row.get("centre_name", code),
                ).add_to(nearby_group)
        nearby_group.add_to(m)
    # ── Route info box ──────────────────────────────────────────────────────
    if route_info:
        rec = route_info.get("Recommendation", "N/A")
        rec_color = PRIMARY if rec == "FTL" else SECONDARY
        info_html = f"""
        <div style="position:fixed;bottom:20px;left:20px;z-index:999;
                    background:rgba(14,17,23,0.9);backdrop-filter:blur(10px);
                    border:1px solid {rec_color}40;border-radius:12px;
                    padding:12px 16px;font-family:Inter,sans-serif;max-width:260px;">
            <div style="font-size:14px;font-weight:700;color:{rec_color};margin-bottom:6px;">
                Recommended: {rec}</div>
            <div style="font-size:11px;color:#8B949E;">
                FTL ETA: {route_info.get('predicted_ETA_FTL', 'N/A')} hrs |
                Carting ETA: {route_info.get('predicted_ETA_Carting', 'N/A')} hrs
            </div>
            <div style="font-size:11px;color:#FFB74D;margin-top:4px;">
                Utility Δ: ₹{route_info.get('Utility_Score_Difference', 'N/A')}
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(info_html))
    folium.LayerControl(collapsed=True).add_to(m)
    # Fit bounds
    m.fit_bounds([[s_lat, s_lng], [d_lat, d_lng]], padding=[40, 40])
    return m

def create_corridor_map(
    coords_df: pd.DataFrame,
    corridor_df: pd.DataFrame,
    top_n: int = 20,
    metric = "median_delay_ratio",
    height: int = 550,
):
    """
    Create a map showing top delayed/breached corridors as colored lines.
    Lines are colored from green (low delay) to red (high delay).
    """
    lookup = lookup_df(coords_df)
    m = folium.Map(
        location=[22.5, 79.0],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    top_corridors = corridor_df.sort_values(by=metric, ascending=False).head(top_n)
    if len(top_corridors) == 0:
        return m
    min_val = top_corridors[metric].min()
    max_val = top_corridors[metric].max()
    val_range = max_val - min_val if max_val > min_val else 1.0
    for _, row in top_corridors.iterrows():
        src = row.get("source_center")
        dst = row.get("destination_center")
        if src not in lookup or dst not in lookup:
            continue
        s_lat, s_lng = lookup[src][0], lookup[src][1]
        d_lat, d_lng = lookup[dst][0], lookup[dst][1]
        # Color interpolation: green → yellow → red
        ratio = (row[metric] - min_val) / val_range
        if ratio < 0.5:
            r, g = int(255 * ratio * 2), 255
        else:
            r, g = 255, int(255 * (1 - ratio) * 2)
        color = f"#{r:02x}{g:02x}00"
        tooltip_text = (
            f"{lookup[src][2]} → {lookup[dst][2]}\n"
            f"Delay Ratio: {row.get('median_delay_ratio', 'N/A'):.2f}x\n"
            f"Breaches: {row.get('breaches', 'N/A')}\n"
            f"Trips: {row.get('total_trips', 'N/A')}"
        )
        folium.PolyLine(
            locations=[[s_lat, s_lng], [d_lat, d_lng]],
            color=color,
            weight=3,
            opacity=0.7,
            tooltip=tooltip_text,
        ).add_to(m)
        # Small markers at endpoints
        for lat, lng, name in [(s_lat, s_lng, lookup[src][2]), (d_lat, d_lng, lookup[dst][2])]:
            folium.CircleMarker(
                location=[lat, lng],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=name,
            ).add_to(m)
    return m
df = load_data()
df = df[df['is_cutoff'] == True].copy()
df.drop(columns=["route_schedule_uuid","cutoff_timestamp","trip_creation_time"],inplace=True)
df['od_start_time'] = pd.to_datetime( df['od_start_time'], format='%d-%m-%Y %H:%M',errors='coerce')
df['hour'] = df['od_start_time'].dt.hour 
df['time_of_day'] =df['hour'].apply(time_bucket)
corridor_df=corridor_setup(df)
print("cooridor df is done ")
map=create_corridor_map(
    coords_df,corridor_df,20,
    "median_delay_ratio",
    550,
)
map.save("C:/Users/lalit/Downloads/ETA_ML_Project/python_files/corridor_map.html")