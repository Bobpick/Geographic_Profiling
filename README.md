# GeoCrime: The Forensic Forecaster

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![OSMnx](https://img.shields.io/badge/OSMnx-v1.x-green.svg)](https://osmnx.readthedocs.io/)

GeoCrime is a Python-based tool designed to aid in forensic investigations by predicting potential future crime locations. It utilizes geographic profiling, machine learning, and data visualization techniques to create a comprehensive map of criminal activity and risk areas.

**By Robert Pickett (c) 2024**

## Features

- **Predictive Algorithms:** Employs multiple machine learning models (Rossmo's Formula, Crime Geographic Targeting, Negative Exponential Distance Decay, Linear Distance Decay) to predict future crime locations.
- **Heatmap Visualization:** Generates a heatmap to visualize the probability distribution of future crime locations based on combined model predictions.
- **Consensus Highlighting:** Identifies and marks locations where multiple models agree on high probability, indicating areas of increased concern.
- **Data Integration:** Incorporates crime data from CSV files and sex offender location data from PDF files.
- **Interactive Map:** Creates an interactive Folium map for easy exploration of crime data, predictions, and locations of interest.
   [![Image 2: Screenshot of the map with a potential next locations][(https://github.com/Bobpick/Geographic_Profiling/blob/main/potential_next_location.png](https://github.com/Bobpick/Geographic_Profiling/blob/main/potential_next_location.png)))
## Getting Started

1. **Prerequisites:** Ensure you have Python 3.x installed along with the required libraries (see `requirements.txt`).

2. **Data Preparation:**
   - **Crime Data (CSV):**
      - File name: `your_crime_data.csv`
      - Columns:
         - `Latitude`: Latitude coordinate of the crime
         - `Longitude`: Longitude coordinate of the crime
         - `Date`: Date of the crime (e.g., "2024-06-22")
         - `Time`: Time of the crime (e.g., "14:30")
         - All on one line, separated by a comma
   - **Sex Offender Data (PDF):**
      - File name: `location.pdf`
      - Format:
         - The PDF should contain individual entries for each sex offender.
         - Each entry must have the following text labels and their corresponding values on separate lines:
            - "First and Last Name:" followed by the offender's name on the next line
            - "Location Details:" followed by the offender's address on the next line

![Location of offenders](Isolated.png "Title")
3. **Installation:**
   - Clone this repository: `git clone https://github.com/your-username/GeoCrime.git`
   - Install dependencies: `pip install -r requirements.txt`
   
4. **Run the Script:**
   - Execute the Python script: `python GeoCrime.py` (Replace "GeoCrime.py" with the actual filename if you renamed it)

## Understanding the Predictive Algorithms

- **Predictive Algorithms:** Employs multiple machine learning models (Rossmo's Formula, Crime Geographic Targeting, Negative Exponential Distance Decay, Linear Distance Decay) to predict future crime locations.
- **Heatmap Visualization:** Generates a heatmap to visualize the probability distribution of future crime locations based on combined model predictions.
- **Consensus Highlighting:** Identifies and marks locations where multiple models agree on high probability, indicating areas of increased concern.
- **Data Integration:** Incorporates crime data from CSV files and sex offender location data from PDF files.
- **Interactive Map:** Creates an interactive Folium map for easy exploration of crime data, predictions, and locations of interest.

The tool combines predictions from all these models, giving equal weight to each, and also highlights areas where multiple models agree.
[![Predicted Locations](https://github.com/Bobpick/Geographic_Profiling/blob/main/potential_next_location.png)(https://github.com/Bobpick/Geographic_Profiling/blob/main/potential_next_location.png)]
## Important Note

This tool is meant to assist in investigations, not replace them. Its predictions are based on statistical models and should not be considered definitive. Always use this information in conjunction with professional judgment and other investigative techniques.

## License

This project is licensed under the Apache 2.0 License.


