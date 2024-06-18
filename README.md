### README.md

# GeoCrime: The Forensic Forecaster
This is still under development!
By Robert Pickett (c) 2024

## Overview

GeoCrime is a Python-based tool designed to assist in crime analysis and prediction. It leverages geographic profiling techniques, machine learning, and open-source data to identify potential crime hotspots and individuals of interest.

## Features

* **Geographic Profiling:** Calculates probabilities of crime occurrence based on various geographic models (CGT, Negative Exponential, Linear Distance Decay).
* **Centrographic Analysis:** Identifies central points of crime clusters (CMD, Mean Center, Median Center).
* **Mapping:** Visualizes crime locations, street networks, and potential areas of interest on an interactive map.
* **Sex Offender Data Integration:** Incorporates sex offender locations from PDF files for enhanced analysis.
* **Machine Learning Prediction:** Trains a model to predict future crime locations based on historical data, time, and celestial events.
* **PDF Report Generation:** Creates a comprehensive report summarizing the analysis findings.

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
   * **Crime_Map_With_Predictions.html:** Interactive map with additional heatmap showing predicted crime probabilities.
   * **Crime_Report.pdf:** PDF report summarizing the analysis.

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
