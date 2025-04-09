# Public Transport Analysis 

Project by Elina Yancheva

This repository contains a data science project analyzing public transportation usage patterns in Sofia, Bulgaria. The project examines various aspects of public transit data including transport type distribution, temporal patterns, transfer analysis, and geospatial insights.

## Project Overview

The analysis explores a dataset of public transport trips in Sofia, revealing insights about:

- Transport type distribution (metro, bus, tram, trolleybus)
- Hourly usage patterns and peak travel times
- Transfer analysis between different modes of transport
- Geospatial patterns using heatmaps
- Commuter vs. occasional user behavior
- Clustering of trip patterns using K-means and DBSCAN

## Repository Structure

- `public-transport.ipynb` - The Jupyter notebook containing all analysis code and findings
- `data/transport_data.csv` - The dataset used for analysis 
- `public-transport.html` - HTML export of the notebook

## Presentation Slides

Auto-generated presentation slides created from the notebook, source visible only in `slideshow` branch. The slideshow is automatically deployed on the following [GitHub Pages link](https://elinayancheva.github.io/public-transport-analysis/)

### Generating Slides

The slides were generated using Jupyter's nbconvert tool. To regenerate the slides from the notebook:

```bash
jupyter nbconvert --to slides --no-input public-transport.ipynb --output docs/index.html --SlidesExporter.reveal_scroll=True
```

## Requirements

To run the analysis notebook, you'll need:

- Python 3.x
- pandas
- matplotlib
- seaborn
- folium
- scikit-learn
- numpy

Detailed requirements information in the `requirements.txt` file. Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository
2. Run the Jupyter notebook to see the full analysis

or 

To view the slides check the [GitHub Pages link](https://elinayancheva.github.io/public-transport-analysis/) or checkout the `slideshow` branch and open the `docs/index.html` file in a browser


## Acknowledgments

- Data provided by Theoremus