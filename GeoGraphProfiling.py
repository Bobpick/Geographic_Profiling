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
        folium.PolyLine(
            locations=[(row["y"], row["x"]), (row["y"], row["x"])],  # Swap x and y for Folium
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

def announce_and_disclaimer():
    """Display the announcement and disclaimer before starting the process."""
    announcement = """
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
        folium.PolyLine(
            locations=[(row["y"], row["x"]), (row["y"], row["x"])],  # Swap x and y for Folium
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
    print(f'ROC-AUC: {roc_auc_score(y_test, y_pred)}')
    
    return model

def predict_crime_locations(model, node_coords, pagan_holidays, moon_phases):
    """Predict crime probabilities for each node in the street network."""
    now = datetime.now()
    dates = [now.strftime('%Y-%m-%d')] * len(node_coords)
    times = [now.strftime('%H:%M:%S')] * len(node_coords)
    df = pd.DataFrame(node_coords, columns=['Latitude', 'Longitude'])
    df['date'] = dates
    df['time'] = times
    
    features = prepare_features(df, pagan_holidays, moon_phases)
    probabilities = model.predict_proba(features)[:, 1]  # Assuming binary classification
    return probabilities

def generate_pdf_report(crime_coords, cmd, mean_center, median_center, model_performance, pagan_holidays, moon_phases, file_name="Crime_Report.pdf"):
    """Generate a PDF report of the crime analysis."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Crime Analysis Report", ln=True, align='C')

    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Crime Coordinates: {crime_coords}", ln=True)
    pdf.cell(200, 10, txt=f"Center of Minimum Distance (CMD): {cmd}", ln=True)
    pdf.cell(200, 10, txt=f"Mean Center: {mean_center}", ln=True)
    pdf.cell(200, 10, txt=f"Median Center: {median_center}", ln=True)
    
    # Model performance metrics
    pdf.cell(200, 10, txt="Model Performance:", ln=True)
    for key, value in model_performance.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    # Pagan holidays and moon phases
    pdf.cell(200, 10, txt="Pagan Holidays:", ln=True)
    for holiday in pagan_holidays:
        pdf.cell(200, 10, txt=f"{holiday}", ln=True)

    pdf.cell(200, 10, txt="Moon Phases:", ln=True)
    for date, phase in moon_phases.items():
        pdf.cell(200, 10, txt=f"{date}: {phase}", ln=True)

    pdf.output(file_name)

def main():
    announce_and_disclaimer()

    crime_coords, crime_data = load_crime_data("your_crime_data.csv")
    if crime_coords is None:
        return

    individual_data = extract_individuals_info("location.pdf")
    individual_data = geocode_addresses(individual_data)
    individuals_coords = individual_data[['Latitude', 'Longitude']].values

    G = ox.graph_from_address("Los Angeles, California, USA", dist=10000, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    node_coords = np.array([(row.y, row.x) for _, row in nodes.iterrows()])

    cmd = center_of_minimum_distance(crime_coords)
    mean_center_point = mean_center(crime_coords)
    median_center_point = median_center(crime_coords)

    m = folium.Map(location=cmd, zoom_start=12)
    add_markers_to_map(crime_coords, cmd, mean_center_point, median_center_point, m)
    add_street_network_to_map(edges, m)
    add_individuals_to_map(individuals_coords, crime_coords, buffer_radius=10000)

    probabilities = crime_geographic_targeting(node_coords, crime_coords)
    add_probability_heatmap(node_coords, probabilities, m, color="red")

    m.save("Crime_Map.html")

    # Define pagan holidays
    norse_pagan_holidays = [
        '2024-12-20', '2024-12-21', '2024-12-22', '2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', '2024-12-28', '2024-12-29', '2024-12-30', '2024-12-31',
        '2024-02-20', '2024-02-21', '2024-02-22', '2024-02-23', '2024-02-24', '2024-02-25', '2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29', '2024-03-01', '2024-03-02', '2024-03-03',
        '2024-03-20', '2024-03-21', '2024-03-22', '2024-03-23',
        '2024-04-30',
        '2024-06-20', '2024-06-21', '2024-06-22', '2024-06-23',
        '2024-08-01',
        '2024-10-29', '2024-10-30', '2024-10-31',
        '2025-12-20', '2025-12-21', '2025-12-22', '2025-12-23', '2025-12-24', '2025-12-25', '2025-12-26', '2025-12-27', '2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31',
        '2025-02-19', '2025-02-20', '2025-02-21', '2025-02-22', '2025-02-23', '2025-02-24', '2025-02-25', '2025-02-26', '2025-02-27', '2025-02-28', '2025-03-01', '2025-03-02', '2025-03-03',
        '2025-03-20', '2025-03-21', '2025-03-22', '2025-03-23',
        '2025-04-30',
        '2025-06-20', '2025-06-21', '2025-06-22', '2025-06-23',
        '2025-08-01',
        '2025-10-29', '2025-10-30', '2025-10-31'
    ]

    african_holidays = [
        '2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04', '2024-08-05', '2024-08-06', '2024-08-07', '2024-08-08', '2024-08-09', '2024-08-10', '2024-08-11', '2024-08-12', '2024-08-13', '2024-08-14', '2024-08-15', '2024-08-16', '2024-08-17', '2024-08-18', '2024-08-19', '2024-08-20', '2024-08-21', '2024-08-22', '2024-08-23', '2024-08-24', '2024-08-25', '2024-08-26', '2024-08-27', '2024-08-28', '2024-08-29', '2024-08-30', '2024-08-31', '2024-09-01',
        '2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04', '2024-08-05', '2024-08-06', '2024-08-07', '2024-08-08', '2024-08-09', '2024-08-10', '2024-08-11', '2024-08-12', '2024-08-13', '2024-08-14', '2024-08-15', '2024-08-16', '2024-08-17', '2024-08-18', '2024-08-19', '2024-08-20', '2024-08-21', '2024-08-22', '2024-08-23', '2024-08-24', '2024-08-25', '2024-08-26', '2024-08-27', '2024-08-28', '2024-08-29', '2024-08-30', '2024-08-31', '2024-09-01',
        '2024-01-19',
        '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10', '2025-08-11', '2025-08-12', '2025-08-13', '2025-08-14', '2025-08-15', '2025-08-16', '2025-08-17', '2025-08-18', '2025-08-19', '2025-08-20', '2025-08-21', '2025-08-22', '2025-08-23', '2025-08-24', '2025-08-25', '2025-08-26', '2025-08-27', '2025-08-28', '2025-08-29', '2025-08-30', '2025-08-31', '2025-09-01',
        '2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05', '2025-08-06', '2025-08-07', '2025-08-08', '2025-08-09', '2025-08-10', '2025-08-11', '2025-08-12', '2025-08-13', '2025-08-14', '2025-08-15', '2025-08-16', '2025-08-17', '2025-08-18', '2025-08-19', '2025-08-20', '2025-08-21', '2025-08-22', '2025-08-23', '2025-08-24', '2025-08-25', '2025-08-26', '2025-08-27', '2025-08-28', '2025-08-29', '2025-08-30', '2025-08-31', '2025-09-01',
        '2025-01-19'
    ]

    haitian_vodou_holidays = [
        '2024-11-01', '2024-11-02',
        '2024-03-17',
        '2024-10-25',
        '2025-11-01', '2025-11-02',
        '2025-03-17',
        '2025-10-25'
    ]

    mexican_holidays = [
        '2024-11-01', '2024-11-02',
        '2024-02-02',
        '2024-06-23',
        '2025-11-01', '2025-11-02',
        '2025-02-02',
        '2025-06-23'
    ]

    # Combine all holidays
    pagan_holidays = norse_pagan_holidays + african_holidays + haitian_vodou_holidays + mexican_holidays
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now() + timedelta(days=30)
    moon_phases = calculate_moon_phases(start_date, end_date)

    model, model_performance = train_model(crime_data, pagan_holidays, moon_phases)

    predicted_probabilities = predict_crime_locations(model, node_coords, pagan_holidays, moon_phases)
    add_probability_heatmap(node_coords, predicted_probabilities, m, color="blue")

    m.save("Crime_Map_With_Predictions.html")

    generate_pdf_report(crime_coords, cmd, mean_center_point, median_center_point, model_performance, pagan_holidays, moon_phases, file_name="Crime_Report.pdf")

if __name__ == "__main__":
    main()
