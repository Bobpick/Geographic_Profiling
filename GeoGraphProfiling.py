import pandas as pd
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import folium
from folium import plugins
from folium.features import CustomIcon
import osmnx as ox
from scipy.spatial.distance import cdist
from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
import numpy as np
import fitz

NOMINATIM_USER_AGENT = "GeoCrime"

def announce_and_disclaimer():
    """Display the announcement and disclaimer before starting the process."""
    announcement = """
                    GeoCrime: The Forensic Forecaster
                    By Robert Pickett (c) 2024

    Please put the crime information with latitude, longitude, date, and time in separate columns and save as "your_crime_data.csv"
    Please make sure sex offender data is saved as "location.pdf" in the same directory.
    """
    disclaimer = """
    DISCLAIMER: In no way is this 100% guaranteed to be correct and is still in the testing phases. It can make mistakes and is no replacement for good detective work.
    """
    print(announcement)
    print(disclaimer)
    input("Press any key to start...")

def load_crime_data(file_path):
    """Loads crime data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Latitude', 'Longitude', 'Date', 'Time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the CSV file: {', '.join(missing_columns)}")
        return df[['Latitude', 'Longitude', 'Date', 'Time']].dropna()
    except Exception as e:
        print(f"Error loading crime data: {e}")
        return None

def crime_geographic_targeting(coords, crime_coords, k=1, f=1.2, g=1.2, buffer_radius=10000):
    """Calculates the CGT probability for each coordinate."""
    coords = np.atleast_2d(coords)
    crime_coords = np.atleast_2d(crime_coords)

    if coords.shape[1] != crime_coords.shape[1]:
        raise ValueError("coords and crime_coords must have the same number of columns")

    distances = cdist(coords, crime_coords)
    phi = 1 / (distances + 1e-8)  # Avoid division by zero
    part1 = k * np.sum(phi**f, axis=1)
    part2 = (1 - phi) * (buffer_radius**g - distances**g) / (2 * buffer_radius - distances)
    part2[distances >= buffer_radius] = 0  # Set to 0 outside buffer
    return part1 + np.sum(part2, axis=1)

def negative_exponential(coords, crime_coords, A=1.89, B=-0.06):
    """Calculates the Negative Exponential probability for each coordinate."""
    coords = np.atleast_2d(coords)
    crime_coords = np.atleast_2d(crime_coords)

    if coords.shape[1] != crime_coords.shape[1]:
        raise ValueError("coords and crime_coords must have the same number of columns")

    distances = cdist(coords, crime_coords)
    return A * np.exp(B * distances)

def linear_distance_decay(coords, crime_coords, A=1.9, B=-0.06):
    """Calculates the Linear Distance Decay probability for each coordinate."""
    coords = np.atleast_2d(coords)
    crime_coords = np.atleast_2d(crime_coords)

    if coords.shape[1] != crime_coords.shape[1]:
        raise ValueError("coords and crime_coords must have the same number of columns")

    distances = cdist(coords, crime_coords)
    return A + B * distances

def calculate_geographic_profile(nodes, crime_coords, buffer_radius=1000, f=1.2):
    """Calculates the geographic profile based on Rossmo's formula."""
    node_coords = nodes[['y', 'x']].values
    distances = cdist(node_coords, crime_coords)
    phi = 1 / (distances + 1e-8)  # Avoid division by zero
    part1 = np.sum(phi**f, axis=1)
    part2 = (1 - phi) * (buffer_radius**f - distances**f) / (2 * buffer_radius - distances)
    part2[distances >= buffer_radius] = 0  # Set to 0 outside buffer
    geographic_profile = part1 + np.sum(part2, axis=1)
    return geographic_profile

def calculate_anchor_point(nodes, geographic_profile):
    """Calculates the anchor point based on the geographic profile."""
    max_idx = np.argmax(geographic_profile)
    return (nodes.iloc[max_idx]['y'], nodes.iloc[max_idx]['x'])

def predict_future_crime_locations_multiple(nodes, crime_coords, buffer_radius=5500, num_locations=5):
    node_coords = nodes[['y', 'x']].values

    probabilities_rossmo = calculate_geographic_profile(nodes, crime_coords, buffer_radius, f=1.5)
    probabilities_cgt = crime_geographic_targeting(node_coords, crime_coords, buffer_radius=buffer_radius)
    probabilities_neg_exp = negative_exponential(node_coords, crime_coords).sum(axis=1)  # Sum probabilities across crimes
    probabilities_linear = linear_distance_decay(node_coords, crime_coords).sum(axis=1)  # Sum probabilities across crimes

    probabilities_rossmo = normalize_probabilities(probabilities_rossmo).reshape(-1, 1)
    probabilities_cgt = normalize_probabilities(probabilities_cgt).reshape(-1, 1)
    probabilities_neg_exp = normalize_probabilities(probabilities_neg_exp).reshape(-1, 1)
    probabilities_linear = normalize_probabilities(probabilities_linear).reshape(-1, 1)

    combined_probabilities = np.hstack([probabilities_rossmo, probabilities_cgt, probabilities_neg_exp, probabilities_linear]).mean(axis=1)

    top_indices = np.argsort(combined_probabilities)[-num_locations:][::-1]
    future_crime_coords = node_coords[top_indices]
    
    models = ['Rossmo\'s Formula', 'CGT', 'Negative Exponential', 'Linear Distance Decay']
    predicted_by = np.array(models)[np.argmax(np.hstack([probabilities_rossmo, probabilities_cgt, probabilities_neg_exp, probabilities_linear]), axis=1)]
    top_predictions = predicted_by[top_indices]
    
    return future_crime_coords, top_predictions

def normalize_probabilities(probabilities):
    prob_min = np.min(probabilities)
    prob_max = np.max(probabilities)
    if prob_max == prob_min:
        return np.zeros_like(probabilities)
    return (probabilities - prob_min) / (prob_max - prob_min)

def add_probability_heatmap(node_coords, probabilities, folium_map, colormap='Reds'):
    normalized_probs = normalize_probabilities(probabilities)
    colormap_func = plt.get_cmap(colormap)
    colors_rgba = colormap_func(normalized_probs)
    colors_rgba[:, 3] = 1.0  # Remove transparency (alpha channel)
    colors_hex = [mcolors.to_hex(rgba) for rgba in colors_rgba]

    for (lat, lon), color in zip(node_coords, colors_hex):
        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0
        ).add_to(folium_map)


def add_markers_to_map(crime_data, folium_map):
    for _, row in crime_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Date: {row['Date']} Time: {row['Time']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(folium_map)

def extract_individuals_info(pdf_path):
    doc = fitz.open(pdf_path)
    individuals = []

    for page in doc:
        text = page.get_text("text")
        lines = text.splitlines()

        for i in range(len(lines)):
            if lines[i].strip() == "First and Last Name":
                name = lines[i + 1].strip()
                if lines[i + 2].strip() == "Location Details":
                    address = lines[i + 3].strip()
                    individuals.append({"name": name, "address": address})
    return pd.DataFrame(individuals)

def geocode_addresses(df):
    geolocator = Nominatim(user_agent="crime_mapping")
    coords = []

    for address in df['address']:
        try:
            location = geolocator.geocode(address)
            if location:
                coords.append((location.latitude, location.longitude))
            else:
                coords.append((None, None))
        except Exception as e:
            print(f"Error geocoding address '{address}': {e}")
            coords.append((None, None))

    df['latitude'] = [lat for lat, _ in coords]
    df['longitude'] = [lon for _, lon in coords]

    return df.dropna(subset=['latitude', 'longitude'])

def create_map_with_legend(crime_data, future_crime_locations, sex_offenders, base_location):
    m = folium.Map(location=base_location, zoom_start=13)

    add_markers_to_map(crime_data, m)

    model_colors = {
        "Rossmo's Formula": 'green',
        "CGT": 'orange',
        "Negative Exponential": 'purple',
        "Linear Distance Decay": 'pink'
    }

    for (lat, lon), model in future_crime_locations:
        color = model_colors.get(model, 'black')
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=color, icon='info-sign'),
            popup=f"Predicted Future Crime Location by {model}"
        ).add_to(m)

    for _, row in sex_offenders.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['name'],
            icon=folium.Icon(color='blue', icon='user')
        ).add_to(m)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 300px; height: 280px; background-color: white; 
    border: 2px solid grey; z-index: 9999; font-size: 14px;">
    <h4 style="margin-top: 10px;">Legend</h4>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li><span style="background-color: red; width: 12px; height: 12px; display: inline-block;"></span> 
        Crime Locations</li>
        <li><span style="background-color: green; width: 12px; height: 12px; display: inline-block;"></span> 
        Predicted Future Crime Locations by Rossmo's Formula</li>
        <li><span style="background-color: orange; width: 12px; height: 12px; display: inline-block;"></span> 
        Predicted Future Crime Locations by CGT</li>
        <li><span style="background-color: purple; width: 12px; height: 12px; display: inline-block;"></span> 
        Predicted Future Crime Locations by Negative Exponential</li>
        <li><span style="background-color: pink; width: 12px; height: 12px; display: inline-block;"></span> 
        Predicted Future Crime Locations by Linear Distance Decay</li>
        <li><span style="background-color: blue; width: 12px; height: 12px; display: inline-block;"></span> 
        Sex Offenders</li>
        <li>
            <span style="background: linear-gradient(to right, pink, red); width: 50px; height: 12px; display: inline-block;"></span>
            Probability of Criminal's Location (50% to 100%)
        </li>
    </ul>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def main():
    announce_and_disclaimer()

    crime_data_file_path = "your_crime_data.csv"
    crime_data = load_crime_data(crime_data_file_path)
    if crime_data is None:
        print("Unable to load crime data. Exiting...")
        return

    location_file_path = "location.pdf"
    sex_offenders = extract_individuals_info(location_file_path)
    sex_offenders = geocode_addresses(sex_offenders)
    base_location = [crime_data['Latitude'].mean(), crime_data['Longitude'].mean()]

    crime_coords = crime_data[['Latitude', 'Longitude']].values

    graph = ox.graph_from_point(base_location, dist=10000, network_type='drive')
    nodes = ox.graph_to_gdfs(graph, edges=False).reset_index()
    node_coords = nodes[['y', 'x']].values

    future_crime_coords, top_predictions = predict_future_crime_locations_multiple(nodes, crime_coords)

    geographic_profile = calculate_geographic_profile(nodes, crime_coords)
    anchor_point = calculate_anchor_point(nodes, geographic_profile)

    future_crime_with_model = list(zip(future_crime_coords, top_predictions))

    folium_map = create_map_with_legend(crime_data, future_crime_with_model, sex_offenders, base_location)
    add_probability_heatmap(node_coords, geographic_profile, folium_map)

    folium_map.save("predicted_crime_locations_map.html")
    print("Map saved as 'predicted_crime_locations_map.html'")

if __name__ == "__main__":
    main()
