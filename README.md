# GeoCrime: The Forensic Forecaster

[![Overview Image](https://github.com/Bobpick/Geographic_Profiling/raw/main/overview_image.png)](https://github.com/Bobpick/Geographic_Profiling/blob/main/overview_image.png)

GeoCrime is a Python-based tool designed to aid criminal investigations by applying geographic profiling techniques. It utilizes crime location data, street network information, and potentially offender data to estimate likely anchor points (e.g., residence or base of operations) and predict future crime locations.

## Features

* **Crime Data Analysis:** Processes crime data from a CSV file containing latitude, longitude, date, and time.
* **Geographic Profiling:** Applies Rossmo's formula and other techniques to generate a geographic profile, highlighting areas with higher probabilities of the offender's residence.
* **Anchor Point Estimation:** Calculates the most likely anchor point based on the geographic profile.
* **Future Crime Prediction:** Predicts potential future crime locations within a buffer zone around the anchor point.
* **Interactive Map Visualization:**  Creates an interactive Folium map that visualizes:
    * **Crime Locations:** Markers for past crime incidents.
    * **Heatmap:** Visual representation of the geographic profile (probability distribution).
    * **Anchor Point:** Estimated location of the offender's base.
    * **Future Crime Predictions:**  Markers for potential future crime sites.
    * [![Image 2: Screenshot of the map with a potential next locations](https://github.com/Bobpick/Geographic_Profiling/blob/main/potential_next_location.png))
    * **Street Network:** Overlay of roads and streets for context.
* **Registered Sex Offender Overlay:**  Option to overlay the locations of registered sex offenders from a PDF file.
[![Image 3: Screenshot of the map with registered offender](https://github.com/Bobpick/Geographic_Profiling/blob/main/zoomed_in_image.png)

* **Address Geocoding:**  Uses the OpenCage API to geocode addresses of individuals for mapping.
* **Customizable Parameters:** Allows adjusting parameters like the buffer zone radius and decay factor for fine-tuning the analysis.

## How to Use

1. **Prepare Input Data:**
   * Create a CSV file named "your_crime_data.csv" with columns: `Latitude`, `Longitude`, `Date`, `Time`.
   * Place the file in the same directory as the script.
   * Optionally, prepare a PDF file named "location.pdf" containing the addresses of individuals of interest.
   
2. **Install Dependencies:**
   ```bash
   pip install pandas numpy folium osmnx matplotlib scipy opencage geopy fitz scikit-learn
   
## Installation

1. **Clone the repository:** `git clone https://github.com/your_username/GeoCrime.git`
2. **Install dependencies:** `pip install -r requirements.txt`

## Usage

1. **Prepare your data:**
   * **Crime Data:** Create a CSV file named "your_crime_data.csv" with columns for Latitude, Longitude, Date, and Time.
   * **Sex Offender Data:** Obtain a PDF file named "location.pdf" containing sex offender information (see example format below).
2. **Run the script:** `python geocrime.py`
3. **View the results:**
   * **Crime_Map.html:** Interactive map displaying crime locations, street network, and individuals of interest.
 
## Example Data Files

### your_crime_data.csv

```
Latitude,Longitude,Date,Time
34.0522,-118.2437,2024-06-15,14:30:00
34.0622,-118.2537,2024-06-16,02:15:00
34.0422,-118.2337,2024-06-17,21:45:00
```

### location.pdf (Excerpt)

```
First and Last Name
John Doe
Location Details
123 Main St, Los Angeles, CA 90001
```

## Disclaimer

This tool is intended for informational and research purposes only. The predictions and analysis are not guaranteed to be 100% accurate and should not be used as the sole basis for decision-making. Always consult with law enforcement professionals and exercise caution when interpreting the results.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License.
