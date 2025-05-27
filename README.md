# Relationship between Extreme Precipitation Duration and Wet-day Frequency

This repository contains the code for the paper "Unraveling the relationship between extreme precipitation duration and wet-day frequency in Global". This research investigates the relationship between extreme precipitation duration and wet-day frequency on a global scale.

## Project Structure

```
.
├── config.py              # Global parameter configuration
├── config_vis.py          # Visualization parameter configuration
├── data_align.py          # Data alignment processing
├── data_era5.py           # ERA5 data processing
├── data_mswep.py          # MSWEP data processing
├── data_persiann.py       # PERSIANN data processing
├── map_lib.py             # Geographic plotting library
├── plot_lib.py            # General plotting library
├── processor_map.py       # Geographic region feature calculation
├── processor_main.py      # Core processing procedures
├── utils.py               # General utility functions
├── utils_event.py         # Precipitation event duration processing tools
├── workflow.py            # Overall workflow
├── visualization_main.py  # Main visualizations
├── visualization_supp.py  # Supplementary visualizations
└── test-other-fit.py      # Other function fitting tests
```

## Requirements

- Python 3.8+
- Key dependencies:
  - numpy
  - pandas
  - xarray
  - netCDF4
  - matplotlib
  - cartopy
  - scipy
  - scikit-learn

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd precipitation-upload
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:

- The ERA5 reanalysis products are obtained from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/).

- MSWEP dataset is available at (https://www.gloh2o.org/mswep/).
- PERSIANN dataset is available at (https://chrsdata.eng.uci.edu/).

2. Run the workflow:

```bash
python workflow.py
```

3. Generate visualizations:

```bash
python visualization_main.py  # Main figures
python visualization_supp.py  # Supplementary figures
```

4. There are some previously processed data in the internal_data folder for reference

## Main Features

- Data preprocessing and alignment
- Extreme precipitation event identification
- Geographic region feature analysis
- Multi-source data comparison analysis
- Visualization generation

## Notes

- Ensure sufficient disk space for data processing
- It is recommended to use a virtual environment
- Some features may require significant memory resources

## Contact

2571854608gl@gmail.com
