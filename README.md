# UFA Stats Dashboard

A data collection and visualization tool for Ultimate Frisbee Association player statistics. This project scrapes player stats from watchufa.com and provides an interactive dashboard for analysis.

## Features

- **Automated Data Scraper**: Collects player statistics from watchufa.com for multiple seasons (2012-2024)
- **Interactive Dashboard**: Visualize and analyze player performance data using Streamlit
- **Filtering Capabilities**: Filter data by season, team, player, and various statistical metrics
- **Data Visualization**: Generate charts and graphs to compare player performance

## Project Structure

- `scrape_ufa.py` - Web scraper for collecting UFA player statistics
- `dashboard.py` - Streamlit dashboard for data visualization and analysis
- `scraped_data/` - Directory containing CSV files with player statistics by season
- `logs/` - Log files from scraping operations
- `debug_output/` - Debug information from scraping sessions

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly
- Selenium
- ChromeDriver
- Other dependencies as listed in the import statements

## Usage

### Data Collection

To scrape player statistics:

```bash
python scrape_ufa.py
```

This will collect data from watchufa.com and save it to CSV files in the `scraped_data/` directory.

### Dashboard

To run the interactive dashboard:

```bash
streamlit run dashboard.py
```

This will start a local web server and open the dashboard in your default browser.

## Data Overview

The collected data includes various statistics for UFA players, such as:
- Games played
- Points
- Possessions
- Scores
- Assists
- Goals
- Blocks
- Completion percentage
- Yards
- Offensive efficiency
- And more

Data is available for seasons from 2012 to 2024 (excluding 2020), as well as career statistics.

## License

MIT License 

## Acknowledgements

- Data sourced from watchufa.com
- Built with Streamlit, Pandas, and Selenium 