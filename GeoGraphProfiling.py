from folium import plugins
from folium.features import customicon
from geopy.exc import geocoderunavailable
from geopy.geocoders import nominatim
from opencage.geocoder import opencagegeocode, ratelimitexceedederror, invalidinputerror, unknownerror
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.ensemble import randomforestregressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kerneldensity
from sklearn.preprocessing import standardscaler
import fitz
import folium
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import osmnx as ox
import pandas as pd
import time

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
        # Check if the required columns are in the dataframe
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
    # Ensure both coords and crime_coords are 2D arrays with the same number of columns
    coords = np.atleast_2d(coords)
    crime_coords = np.atleast_2d(crime_coords)

    print(f"coords shape: {coords.shape}")
    print(f"crime_coords shape: {crime_coords.shape}")

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
    # Ensure both coords and crime_coords are 2D arrays with the same number of columns
    coords = np.atleast_2d(coords)
    crime_coords = np.atleast_2d(crime_coords)

    if coords.shape[1] != crime_coords.shape[1]:
        raise ValueError("coords and crime_coords must have the same number of columns")

    distances = cdist(coords, crime_coords)
    return A * np.exp(B * distances)

def linear_distance_decay(coords, crime_coords, A=1.9, B=-0.06):
    """Calculates the Linear Distance Decay probability for each coordinate."""
    # Ensure both coords and crime_coords are 2D arrays with the same number of columns
    coords = np.atleast_2d(coords)
    crime_coords = np.atleast_2d(crime_coords)

    if coords.shape[1] != crime_coords.shape[1]:
        raise ValueError("coords and crime_coords must have the same number of columns")

    distances = cdist(coords, crime_coords)
    return A + B * distances


def calculate_cmd(crime_coords):
    """Calculates the CMD from crime coordinates."""
    total_distances = np.sum(cdist(crime_coords, crime_coords), axis=1)
    return crime_coords[np.argmin(total_distances)]

def calculate_mean_center(crime_coords):
    """Calculates the Mean Center from crime coordinates."""
    return np.mean(crime_coords, axis=0)

def calculate_median_center(crime_coords):
    """Calculates the Median Center from crime coordinates."""
    return np.median(crime_coords, axis=0)

def add_street_network_to_map(nodes, edges, crime_coords, future_crime_coords, m, street_network_opacity=1.0):
    """Add street network to the Folium map."""
    for _, row in nodes.iterrows():
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=2,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=street_network_opacity,
        ).add_to(m)

    for _, row in edges.iterrows():
        points = row['geometry'].xy
        coord_pairs = list(zip(points[1], points[0]))  # Extract coordinates from geometry
        folium.PolyLine(coord_pairs, color='blue', weight=2.5, opacity=street_network_opacity).add_to(m)

    # Use a custom icon for crime location markers
    icon_path = 'skull.png'  # Ensure this path is correct and the file is in the same directory
    if not os.path.isfile(icon_path):
        print(f"Icon file {icon_path} not found.")
        return

    for coord in crime_coords:
        folium.Marker(
            location=[coord[0], coord[1]],
            popup='Crime Location',
            icon=folium.CustomIcon(icon_image=icon_path, icon_size=(30, 30))  # Adjust the icon size as needed
        ).add_to(m)
        
    # Add green markers for future crime locations
    for coord in future_crime_coords:
        folium.Marker(
            location=[coord[0], coord[1]],
            popup='Future Crime Location',
            icon=folium.Icon(color='green')
        ).add_to(m)


def add_proportional_symbol_markers(crime_data, folium_map):
    """Adds proportional symbol markers to the Folium map."""
    for index, row in crime_data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,  # Adjust radius based on crime count or another attribute
            color='red',
            fill=True,
            fill_color='red',
            popup=f"Date: {row['Date']} Time: {row['Time']}"
        ).add_to(folium_map)
        
def create_folium_map(center_point):
    """Creates a Folium map centered at the given coordinates."""
    return folium.Map(location=center_point, zoom_start=12)

def extract_individuals_info(pdf_path):
    """Extracts individuals' names and addresses from the PDF."""
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
    """Geocodes addresses to coordinates."""
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
            print(f"Error geocoding address {address}: {e}")
            coords.append((None, None))

    df['Latitude'], df['Longitude'] = zip(*coords)
    return df.dropna(subset=['Latitude', 'Longitude'])

def add_markers_to_map(crime_data, mean_center, median_center, m):
    """Add markers for crime locations to the folium map."""
    for index, row in crime_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Date: {row['Date']} Time: {row['Time']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)


def normalize_probabilities(probabilities):
    prob_min = np.min(probabilities)
    prob_max = np.max(probabilities)
    if prob_max == prob_min:
        return np.zeros_like(probabilities)
    return (probabilities - prob_min) / (prob_max - prob_min)


def add_probability_heatmap(node_coords, probabilities, folium_map, colormap='Blues'):
    normalized_probs = normalize_probabilities(probabilities)
    colormap_func = plt.get_cmap(colormap)
    colors_rgba = colormap_func(normalized_probs)

    # Ensure RGBA values are clipped to [0, 1] range and reshape if necessary
    colors_rgba_clipped = np.clip(colors_rgba, 0.0, 1.0).reshape(-1, 4)

    # Convert RGBA to Hex
    colors_hex = [mcolors.to_hex(rgba) for rgba in colors_rgba_clipped]

    # Add circles to the map
    for (lat, lon), color in zip(node_coords, colors_hex):
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(folium_map)


def add_street_network_to_map(nodes, edges, crime_coords, future_crime_coords, m, street_network_opacity=1.0):
    """Add street network to the Folium map."""
    for _, row in nodes.iterrows():
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=2,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=street_network_opacity,
        ).add_to(m)

    for _, row in edges.iterrows():
        points = row['geometry'].xy
        coord_pairs = list(zip(points[1], points[0]))  # Extract coordinates from geometry
        folium.PolyLine(coord_pairs, color='blue', weight=2.5, opacity=street_network_opacity).add_to(m)

    # Use a custom icon for crime location markers
    icon_path = 'skull.png'  # Ensure this path is correct and the file is in the same directory
    if not os.path.isfile(icon_path):
        print(f"Icon file {icon_path} not found.")
        return

    for coord in crime_coords:
        folium.Marker(
            location=[coord[0], coord[1]],
            popup='Crime Location',
            icon=folium.CustomIcon(icon_image=icon_path, icon_size=(30, 30))  # Adjust the icon size as needed
        ).add_to(m)
        
    # Add green markers for future crime locations
    for coord in future_crime_coords:
        folium.Marker(
            location=[coord[0], coord[1]],
            popup='Future Crime Location',
            icon=folium.Icon(color='green')
        ).add_to(m)


def add_individuals_to_map(individuals_coords, crime_coords, buffer_radius, m):
    """Adds individuals' markers to the map if they are within buffer_radius of any crime coordinate."""
    for coord in crime_coords:
        for ind_coord in individuals_coords:
            distance = np.linalg.norm(np.array(coord) - np.array(ind_coord))
            if distance <= buffer_radius:
                folium.Marker(ind_coord, popup="Individual of Interest").add_to(m)


NOMINATIM_USER_AGENT = "GeoCrime"
OPENCAGE_API_KEY = "xxxxxeee86d40b5b196b99d8de2efe9"

def reverse_geocode(coord, timeout=10, max_retries=3):
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
    backup_geocoder = OpenCageGeocode(OPENCAGE_API_KEY)

    retries = 0
    while retries < max_retries:
        try:
            location = geolocator.reverse(coord, timeout=timeout)
            return location.address
        except (GeocoderUnavailable, RateLimitExceededError):
            retries += 1
            time.sleep(1)
        except (InvalidInputError, UnknownError) as e:
            print(f"Error reverse geocoding: {e}")
            return "Address not found"

    try:
        location = backup_geocoder.reverse_geocode(coord[0], coord[1])[0]
        return location['formatted']
    except Exception as e:
        print(f"Backup geocoder failed: {e}")
        return "Address not found"

# Define functions for geographic profiling and prediction
def calculate_geographic_profile(nodes, crime_coords, buffer_radius=1000, f=1.2):
    """
    Calculates the geographic profile based on Rossmo's formula.

    Parameters:
    - nodes: DataFrame containing geographic coordinates of nodes.
    - crime_coords: 2D numpy array of crime coordinates.
    - buffer_radius: Buffer zone radius around anchor point.
    - f: Exponential decay factor.

    Returns:
    - geographic_profile: 1D numpy array of geographic profile values.
    """
    node_coords = nodes[['y', 'x']].values  # Extract node coordinates

    distances = cdist(node_coords, crime_coords)
    phi = 1 / (distances + 1e-8)  # Avoid division by zero
    part1 = np.sum(phi**f, axis=1)
    part2 = (1 - phi) * (buffer_radius**f - distances**f) / (2 * buffer_radius - distances)
    part2[distances >= buffer_radius] = 0  # Set to 0 outside buffer
    geographic_profile = part1 + np.sum(part2, axis=1)

    return geographic_profile


def calculate_anchor_point(nodes, geographic_profile):
    """
    Calculates the anchor point based on the geographic profile.

    Parameters:
    - nodes: DataFrame containing geographic coordinates of nodes.
    - geographic_profile: 1D numpy array of geographic profile values.

    Returns:
    - anchor_point: Tuple of latitude and longitude of the anchor point.
    """
    max_idx = np.argmax(geographic_profile)
    anchor_point = (nodes.iloc[max_idx]['y'], nodes.iloc[max_idx]['x'])

    return anchor_point

def predict_future_crime_locations(anchor_point, buffer_radius, num_locations=3):
    """
    Predicts future crime locations around the anchor point within a buffer zone.

    Parameters:
    - anchor_point: Tuple of latitude and longitude of the anchor point.
    - buffer_radius: Radius of the buffer zone around the anchor point.
    - num_locations: Number of future crime locations to predict.

    Returns:
    - future_crime_coords: List of tuples containing predicted future crime coordinates.
    """
    future_crime_coords = []
    for _ in range(num_locations):
        lat_offset = np.random.uniform(-1, 1) * buffer_radius / 111000  # Convert meters to degrees latitude
        lon_offset = np.random.uniform(-1, 1) * buffer_radius / (111000 * np.cos(np.deg2rad(anchor_point[0])))  # Convert meters to degrees longitude
        future_coord = (anchor_point[0] + lat_offset, anchor_point[1] + lon_offset)
        future_crime_coords.append(future_coord)
    
    return future_crime_coords

def main():
    # Load crime data and extract coordinates
    crime_data = pd.read_csv("your_crime_data.csv")
    crime_coords = crime_data[['Latitude', 'Longitude']].values

    # Load street network around crime locations
    mean_center_point = np.mean(crime_coords, axis=0)
    graph = ox.graph_from_point(mean_center_point, dist=2000, network_type='drive')
    nodes, _ = ox.graph_to_gdfs(graph)

    # Calculate geographic profile
    geographic_profile = calculate_geographic_profile(nodes, crime_coords)

    # Calculate anchor point
    anchor_point = calculate_anchor_point(nodes, geographic_profile)

    # Predict future crime locations based on anchor point and buffer radius
    buffer_radius = 1500  # Example buffer radius in meters
    future_crime_coords = predict_future_crime_locations(anchor_point, buffer_radius)

    # Initialize Folium map centered at anchor point
    m = folium.Map(location=anchor_point, zoom_start=13)

    # Add crime markers to map
    for index, row in crime_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Date: {row['Date']} Time: {row['Time']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # Add street network to map
    for _, row in nodes.iterrows():
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=2,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=0.6,
        ).add_to(m)

    # Add heatmap based on geographic profile
    colormap_func = mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
    colors_rgba = [colormap_func(geographic_profile[i] / np.max(geographic_profile)) for i in range(len(nodes))]
    colors_hex = [mcolors.to_hex(rgba) for rgba in colors_rgba]

    for (idx, color) in zip(nodes.index, colors_hex):
        folium.CircleMarker(
            location=[nodes.loc[idx, 'y'], nodes.loc[idx, 'x']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(m)

    # Add markers for predicted future crime locations
    for coord in future_crime_coords:
        folium.Marker(
            location=[coord[0], coord[1]],
            popup='Future Crime Location',
            icon=folium.Icon(color='green')
        ).add_to(m)

    # Save map as HTML
    m.save("crime_map.html")
    print("Map has been saved as crime_map.html")

if __name__ == "__main__":
    main()
