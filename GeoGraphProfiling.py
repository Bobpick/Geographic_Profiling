import numpy as np
import pandas as pd
import folium
from fpdf import FPDF
import osmnx as ox
from scipy.spatial.distance import cdist
from shapely.geometry import Point, LineString
from geopy.geocoders import Nominatim
import fitz  # PyMuPDF
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import ephem  # For moon phases
from datetime import datetime, timedelta


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


def crime_geographic_targeting(coords, crime_coords, k=1, f=1.2, g=1.2, buffer_radius=10000):
    """Calculates the CGT probability for each coordinate."""
    distances = cdist(coords, crime_coords)
    phi = 1 / (distances + 1e-8)  # Avoid division by zero
    part1 = k * np.sum(phi**f, axis=1)
    part2 = (1 - phi) * (buffer_radius**g - distances**g) / (2 * buffer_radius - distances)
    part2[distances >= buffer_radius] = 0  # Set to 0 outside buffer
    return part1 + np.sum(part2, axis=1)


def negative_exponential(coords, crime_coords, A=1.89, B=-0.06):
    """Calculates the Negative Exponential probability for each coordinate."""
    distances = cdist(coords, crime_coords)
    return A * np.exp(B * distances)


def linear_distance_decay(coords, crime_coords, A=1.9, B=-0.06):
    """Calculates the Linear Distance Decay probability for each coordinate."""
    distances = cdist(coords, crime_coords)
    return A + B * distances


def center_of_minimum_distance(crime_coords):
    """Calculates the CMD from crime coordinates."""
    total_distances = np.sum(cdist(crime_coords, crime_coords), axis=1)
    return crime_coords[np.argmin(total_distances)]


def mean_center(crime_coords):
    """Calculates the Mean Center from crime coordinates."""
    return np.mean(crime_coords, axis=0)


def median_center(crime_coords):
    """Calculates the Median Center from crime coordinates."""
    return np.median(crime_coords, axis=0)


def load_crime_data(file_path):
    """Loads crime data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df[["Latitude", "Longitude"]].values, df  # Assuming your data has these columns
    except Exception as e:
        print(f"Error loading crime data: {e}")
        return None, None


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


def add_markers_to_map(crime_coords, cmd, mean_center, median_center, m):
    """Adds markers to the map for crime locations and centrographic points."""
    # Add markers for crime locations
    for coord in crime_coords:
        address = reverse_geocode(coord)
        folium.Marker(coord, popup=address).add_to(m)
    
    # Add markers for centrographic points
    folium.Marker(cmd, popup="CMD").add_to(m)
    folium.Marker(mean_center, popup="Mean Center").add_to(m)
    folium.Marker(median_center, popup="Median Center").add_to(m)


def add_probability_heatmap(node_coords, probabilities, m, color="red"):
    """Adds a heatmap to the map based on probabilities."""
    for node, prob in zip(node_coords, probabilities):
        folium.CircleMarker(
            location=node,
            radius=prob * 5,  # Scale radius based on probability
            color=color,
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)


def add_street_network_to_map(edges, m):
    """Adds the street network to the map."""
    for _, row in edges.iterrows():
        start_coord = (nodes.loc[row['u'], 'lat'], nodes.loc[row['u'], 'lon'])
        end_coord = (nodes.loc[row['v'], 'lat'], nodes.loc[row['v'], 'lon'])
        folium.PolyLine(
            locations=[start_coord, end_coord],  # Use 'u' and 'v' to get coordinates from nodes
            color="blue",
            weight=1,
            opacity=1,
        ).add_to(m)


def add_individuals_to_map(individuals_coords, crime_coords, buffer_radius, m):
    """Adds individuals' markers to the map if they are within buffer_radius of any crime coordinate."""
    for coord in crime_coords:
        for ind_coord in individuals_coords:
            distance = np.linalg.norm(np.array(coord) - np.array(ind_coord))
            if distance <= buffer_radius:
                folium.Marker(ind_coord, popup="Individual of Interest").add_to(m)


def reverse_geocode(coord):
    """Reverse geocodes coordinates to an address."""
    geolocator = Nominatim(user_agent="crime_mapping")
    location = geolocator.reverse(coord)
    return location.address if location else "Unknown location"


def calculate_moon_phases(start_date, end_date):
    """Calculate moon phases for a range of dates."""
    moon_phases = {}
    current_date = start_date
    while current_date <= end_date:
        ephem_date = ephem.Date(current_date)
        moon = ephem.Moon(ephem_date)
        phase = moon.phase  # Moon phase as a percentage of the moon's illumination
        if phase == 0:
            phase_name = 'New Moon'
        elif phase < 50:
            phase_name = 'First Quarter'
        elif phase == 50:
            phase_name = 'Full Moon'
        else:
            phase_name = 'Last Quarter'
        moon_phases[current_date.strftime('%Y-%m-%d')] = phase_name
        current_date += timedelta(days=1)
    return moon_phases


def prepare_features(df, pagan_holidays, moon_phases):
    """Prepare features for machine learning model."""
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_pagan_holiday'] = df['date'].apply(lambda x: 1 if x in pagan_holidays else 0)
    df['moon_phase'] = df['datetime'].apply(lambda x: moon_phases[x.strftime('%Y-%m-%d')])
    
    features = df[['Latitude', 'Longitude', 'hour', 'day_of_week', 'is_pagan_holiday', 'moon_phase']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled


def train_model(crime_data, pagan_holidays, moon_phases):
    """Train a machine learning model to predict crime locations."""
    features = prepare_features(crime_data, pagan_holidays, moon_phases)
    labels = crime_data['crime_occurred']  # Assuming this is your label column
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred)}')
    
    return model


def create_map():
    """Create an empty Folium map centered on a given location."""
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    return m


def save_map(m, output_file):
    """Save the map as an HTML file."""
    m.save(output_file)


if __name__ == "__main__":
    announce_and_disclaimer()
    
    # Load crime data
    crime_coords, crime_data = load_crime_data("your_crime_data.csv")
    
    if crime_coords is None:
        print("Failed to load crime data. Exiting.")
        exit(1)
    
    # Extract individuals' information from PDF
    individuals_info = extract_individuals_info("location.pdf")
    individuals_info = geocode_addresses(individuals_info)
    
    # Calculate geographic features
    cmd = center_of_minimum_distance(crime_coords)
    mean_center = mean_center(crime_coords)
    median_center = median_center(crime_coords)
    
    # Create map and add features
    m = create_map()
    add_markers_to_map(crime_coords, cmd, mean_center, median_center, m)
    
    # Get street network
    G = ox.graph_from_point((latitude, longitude), dist=3000, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G)
    
    add_street_network_to_map(edges, m)
    
    # Get probabilities
    cgt_probs = crime_geographic_targeting(nodes[['y', 'x']].values, crime_coords)
    ne_probs = negative_exponential(nodes[['y', 'x']].values, crime_coords)
    ldd_probs = linear_distance_decay(nodes[['y', 'x']].values, crime_coords)
    
    add_probability_heatmap(nodes[['y', 'x']].values, cgt_probs, m, color="red")
    add_probability_heatmap(nodes[['y', 'x']].values, ne_probs, m, color="green")
    add_probability_heatmap(nodes[['y', 'x']].values, ldd_probs, m, color="blue")
    
    # Add individuals' locations within buffer radius
    buffer_radius = 10000  # Example buffer radius in meters
    add_individuals_to_map(individuals_info[['Latitude', 'Longitude']].values, crime_coords, buffer_radius, m)
    
    # Save the map
    save_map(m, "crime_map.html")
    
    # Calculate moon phases
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    moon_phases = calculate_moon_phases(start_date, end_date)
    
    # Pagan holidays (example)
    pagan_holidays = ["2023-02-02", "2023-04-30", "2023-06-21", "2023-08-01", "2023-10-31", "2023-12-21"]
    
    # Train the machine learning model
    model = train_model(crime_data, pagan_holidays, moon_phases)
    
    print("Map created and model trained successfully.")
