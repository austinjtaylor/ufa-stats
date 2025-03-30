import pandas as pd
import time
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import logging
from datetime import datetime

# --- Configuration ---
BASE_URL = "https://watchufa.com/stats/player-stats"
MAX_PAGES = 250  # Increased from 20 to handle Career stats with 180 pages
TEST_PAGES = 2  # Number of pages to test per season
WAIT_TIME = 15  # Increased wait time in seconds
INITIAL_WAIT = 3  # Initial wait for page to load
MAX_RETRIES = 3  # Maximum number of times to retry loading a page

# Available filter options based on the image
SEASON_OPTIONS = ["Career"] + [str(year) for year in range(2012, 2025) if year != 2020]
PER_OPTIONS = ["Total", "Per Game", "Per 10 Pts", "Per 10 Pos", "Per 100 Min"]
TEAM_OPTIONS = ["All"]  # Will be populated dynamically

# Available seasons to scrape - Skip 2020 (COVID year)
SEASONS = ["Career"] + [str(year) for year in range(2012, 2025) if year != 2020]

# --- Setup Logging ---
def setup_logging(log_dir="logs"):
    """Configure logging to file and console"""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"scraper_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file

# --- Scraping Function ---
def scrape_ufa_stats(max_pages=MAX_PAGES, start_page=1, season="Career", per="Total", 
                     team="All", testing_mode=False):
    """
    Scrape UFA player stats with the given filter options
    
    Args:
        max_pages: Maximum number of pages to scrape
        start_page: Page to start scraping from
        season: Season filter value (e.g., "Career", "2024", "2023", etc.)
        per: Per filter value (e.g., "Total", "Per Game", etc.) - always uses "Total"
        team: Team filter value (Default: "All")
        testing_mode: If True, only scrape TEST_PAGES from each season
    
    Returns:
        DataFrame or dict with scraped data
    """
    all_player_data = []
    headers = []
    
    # Create debug directory for this scrape session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join("debug_output", f"scrape_{season}_{per}_{timestamp}")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Setup WebDriver
    logging.info("Setting up Chrome WebDriver...")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        # Start with a direct visit to the stats page and set filters
        logging.info(f"Visiting stats page with filters - Season: {season}, Per: {per}, Team: {team}")
        driver.get(BASE_URL)
        time.sleep(INITIAL_WAIT)
        
        # Set filters using dropdown selectors
        filters_set = True
        
        # Set Season filter
        if season != "Career":  # Career is usually the default
            try:
                season_select = WebDriverWait(driver, WAIT_TIME).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 
                        "select#edit-season, select[name*='season']"))
                )
                Select(season_select).select_by_visible_text(season)
                logging.info(f"Set Season filter to: {season}")
            except Exception as e:
                logging.warning(f"Could not set Season filter: {str(e)[:100]}...")
                filters_set = False
        
        # Set Per filter 
        if per != "Total":  # Total is usually the default
            try:
                per_select = WebDriverWait(driver, WAIT_TIME).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 
                        "select#edit-per, select[name*='per']"))
                )
                Select(per_select).select_by_visible_text(per)
                logging.info(f"Set Per filter to: {per}")
            except Exception as e:
                logging.warning(f"Could not set Per filter: {str(e)[:100]}...")
                filters_set = False
        
        # Set Team filter
        if team != "All":  # All is usually the default
            try:
                team_select = WebDriverWait(driver, WAIT_TIME).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 
                        "select#edit-team, select[name*='team']"))
                )
                Select(team_select).select_by_visible_text(team)
                logging.info(f"Set Team filter to: {team}")
            except Exception as e:
                logging.warning(f"Could not set Team filter: {str(e)[:100]}...")
                filters_set = False
        
        # Click apply button if filters were set
        if filters_set:
            try:
                apply_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, 
                        "input[type='submit'][value='Go'], button.form-submit"))
                )
                apply_button.click()
                logging.info("Applied filters")
                time.sleep(2)  # Wait for filters to apply
            except Exception as e:
                logging.warning(f"Could not apply filters: {str(e)[:100]}...")
        else:
            logging.info("Continuing with available filters")
        
        # Limit pages for testing mode
        if testing_mode:
            max_pages = min(max_pages, TEST_PAGES)
            logging.info(f"Testing mode: Will process up to {max_pages} pages per season")
        
        # Process each page and use direct URL navigation only
        current_page = start_page
        page_limit_reached = False
        last_url = None  # Store the last URL to detect when we're stuck on the same page
        repeat_count = 0  # Count how many times we get the same page
        
        while current_page <= max_pages and not page_limit_reached:
            logging.info(f"Processing page {current_page}")
            page_data = []
            page_has_data = False
            
            # Always use direct URL navigation
            retry_count = 0
            load_successful = False
            
            while retry_count < MAX_RETRIES and not load_successful:
                try:
                    # Construct URL based on season - always use direct URL format
                    if season != "Career":
                        page_url = f"{BASE_URL}?page={current_page}&year={season}"
                    else:
                        page_url = f"{BASE_URL}?page={current_page}"
                        
                    logging.info(f"Navigating to URL: {page_url}")
                    driver.get(page_url)
                    time.sleep(3 + retry_count)  # Increase wait time with each retry
                    
                    # Log current URL
                    current_url = driver.current_url
                    logging.info(f"Current URL: {current_url}")
                    
                    # Check if we're stuck on the same page (URL isn't changing)
                    if last_url and current_url == last_url and "page=" in current_url:
                        repeat_count += 1
                        if repeat_count >= 2:  # Allow one repeat in case of temporary issue
                            logging.info(f"Detected page redirection loop - reached the last page for {season}")
                            page_limit_reached = True
                            break
                    else:
                        repeat_count = 0  # Reset counter if URL changes
                    
                    last_url = current_url  # Update last URL
                    
                    # Check if page loaded successfully by looking for content
                    body_text = driver.find_element(By.TAG_NAME, "body").text
                    if "error" in body_text.lower() and "404" in body_text:
                        raise Exception("Page error detected (404)")
                    
                    # Check for tables
                    content_check = driver.find_elements(By.TAG_NAME, "table")
                    if not content_check:
                        content_check = driver.find_elements(By.CSS_SELECTOR, ".view-content, .views-row, .dataTables_wrapper")
                    
                    if not content_check:
                        logging.warning(f"Page seems empty (retry {retry_count+1}/{MAX_RETRIES})")
                        raise Exception("Page appears to have no content")
                    
                    # If we get here, loading was successful
                    load_successful = True
                    
                except Exception as e:
                    retry_count += 1
                    logging.warning(f"Failed to load page {current_page} (attempt {retry_count}/{MAX_RETRIES}): {str(e)[:100]}...")
                    if retry_count < MAX_RETRIES:
                        logging.info(f"Retrying page {current_page}...")
                        time.sleep(2 * retry_count)  # Exponential backoff
                    else:
                        logging.error(f"All {MAX_RETRIES} attempts to load page {current_page} failed, moving to next page")
            
            # Skip further processing if we detected a redirection loop
            if page_limit_reached:
                logging.info(f"Stopping pagination at page {current_page} for {season}")
                break
            
            if not load_successful:
                logging.error(f"Could not load page {current_page} after {MAX_RETRIES} attempts, skipping to next page")
                current_page += 1
                continue
            
            # Save page source for debugging (only in debug level)
            page_source_file = os.path.join(debug_dir, f"page_{current_page}_selenium.html")
            with open(page_source_file, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            
            # Trigger any lazy-loading content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # Try to click any load buttons that might reveal tables
            try:
                load_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'Load')]")
                if load_buttons:
                    logging.info(f"Found {len(load_buttons)} load buttons, attempting to click")
                    for button in load_buttons:
                        try:
                            if button.is_displayed() and button.is_enabled():
                                button.click()
                                time.sleep(2)
                        except Exception:
                            pass
            except Exception:
                pass
            
            # Look for tables using various strategies
            tables = []
            found_table = False
            
            # Strategy 1: Direct table elements
            tables = driver.find_elements(By.TAG_NAME, "table")
            if tables:
                found_table = True
                logging.info(f"Found {len(tables)} tables on the page")
            
            # Strategy 2: Check iframes for tables
            if not found_table or not tables:
                iframes = driver.find_elements(By.TAG_NAME, "iframe")
                if iframes:
                    logging.info(f"Found {len(iframes)} iframes, checking for tables")
                    for i, iframe in enumerate(iframes):
                        try:
                            driver.switch_to.frame(iframe)
                            tables_in_iframe = driver.find_elements(By.TAG_NAME, "table")
                            if tables_in_iframe:
                                tables = tables_in_iframe
                                found_table = True
                                logging.info(f"Found {len(tables)} tables in iframe {i+1}")
                                break
                        except Exception:
                            pass
                        finally:
                            driver.switch_to.default_content()
            
            # Strategy 3: Look for alternative containers that might hold tables
            if not found_table or not tables:
                logging.info("Trying alternative table selectors")
                alt_selectors = [
                    ".stats-table", 
                    "div[id*='table']",
                    ".table-responsive",
                    ".dataTables_wrapper",
                    "div[class*='table']",
                    "div[id*='stats']",
                    "div[class*='stats']",
                    # Add additional selectors that might contain tables
                    ".view-content table",
                    ".views-table",
                    ".view-player-stats",
                    "#player-stats-table",
                    "[data-view-id]"
                ]
                
                for selector in alt_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for element in elements:
                            tables_in_element = element.find_elements(By.TAG_NAME, "table")
                            if tables_in_element:
                                tables = tables_in_element
                                found_table = True
                                logging.info(f"Found {len(tables)} tables with selector {selector}")
                                break
                        if found_table:
                            break
                    except Exception:
                        continue
            
            # If no tables found, try to reload the page one more time as a last resort
            if (not found_table or not tables) and retry_count < MAX_RETRIES:
                logging.warning(f"No tables found on page {current_page}, trying one final reload")
                driver.refresh()
                time.sleep(5)  # Extended wait time after refresh
                
                # Check again for tables after refresh
                tables = driver.find_elements(By.TAG_NAME, "table")
                if tables:
                    found_table = True
                    logging.info(f"Found {len(tables)} tables after page refresh")
            
            # If still no tables found, take screenshot and check if we should continue
            if not found_table or not tables:
                logging.warning(f"No tables found on page {current_page}")
                
                # Take a screenshot for debugging
                screenshot_file = os.path.join(debug_dir, f"page_{current_page}_screenshot.png")
                driver.save_screenshot(screenshot_file)
                
                # Save page text for analysis
                body_text = driver.find_element(By.TAG_NAME, "body").text
                text_file = os.path.join(debug_dir, f"page_{current_page}_text.txt")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(body_text)
                
                # Check for end-of-data indicators
                if "No results found" in body_text or "No matching records found" in body_text:
                    logging.info("Reached end of data (no results message found)")
                    break
                
                current_page += 1
                continue
            
            # Find the stats table among multiple tables
            stats_table = None
            for i, table in enumerate(tables):
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                if len(rows) <= 1:  # Skip tables with only a header row or less
                    continue
                
                # Check header for player stats keywords
                header_cells = rows[0].find_elements(By.TAG_NAME, "th")
                if not header_cells:
                    header_cells = rows[0].find_elements(By.TAG_NAME, "td")
                
                header_text = [cell.text.strip() for cell in header_cells if cell.text.strip()]
                header_joined = ' '.join(header_text).upper()
                
                # Keywords that indicate this is a player stats table
                stats_keywords = ['PLAYER', 'GOALS', 'ASSISTS', 'GLS', 'AST']
                if len(header_text) >= 5 and any(keyword in header_joined for keyword in stats_keywords):
                    stats_table = table
                    logging.info(f"Selected table {i+1} as stats table")
                    break
            
            # Fallback to first multi-row table if no stats table identified
            if not stats_table and tables:
                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if len(rows) > 1:
                        stats_table = table
                        logging.info("Using first multi-row table as fallback")
                        break
            
            if not stats_table:
                logging.warning(f"No suitable table on page {current_page}")
                current_page += 1
                continue
            
            # Extract headers from the table
            extracted_headers = []
            header_cells = []
            
            # Try to find header in thead section first
            thead = stats_table.find_elements(By.TAG_NAME, "thead")
            if thead:
                thead_rows = thead[0].find_elements(By.TAG_NAME, "tr")
                if thead_rows:
                    header_cells = thead_rows[0].find_elements(By.TAG_NAME, "th")
                    if not header_cells:
                        header_cells = thead_rows[0].find_elements(By.TAG_NAME, "td")
            
            # If no thead, use first row
            if not header_cells:
                rows = stats_table.find_elements(By.TAG_NAME, "tr")
                if rows:
                    header_cells = rows[0].find_elements(By.TAG_NAME, "th")
                    if not header_cells:
                        header_cells = rows[0].find_elements(By.TAG_NAME, "td")
            
            # Process headers if found
            if header_cells:
                for cell in header_cells:
                    header_text = cell.text.strip()
                    # Clean up text
                    header_text = re.sub(r'\s+', ' ', header_text)
                    header_text = re.sub(r'[▲▼]', '', header_text).strip()
                    extracted_headers.append(header_text)
                
                if len(extracted_headers) > 5:
                    logging.info(f"Found {len(extracted_headers)} headers")
                    # Update global headers if these seem better
                    if not headers or len(extracted_headers) > len(headers):
                        headers = extracted_headers
                else:
                    logging.warning(f"Found fewer headers than expected: {len(extracted_headers)}")
            
            # Extract data rows
            data_rows = []
            
            # Try to find rows in tbody section first
            tbody = stats_table.find_elements(By.TAG_NAME, "tbody")
            if tbody:
                data_rows = tbody[0].find_elements(By.TAG_NAME, "tr")
                logging.info(f"Found {len(data_rows)} rows in tbody")
            else:
                # No tbody, use all rows except header
                rows = stats_table.find_elements(By.TAG_NAME, "tr")
                data_rows = rows[1:] if rows else []
                logging.info(f"Using {len(data_rows)} rows after header")
            
            if not data_rows:
                logging.warning(f"No data rows on page {current_page}")
                current_page += 1
                continue
            
            # Process data rows
            page_data = []
            page_has_data = False
            
            for row in data_rows:
                # Extract cells
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:
                    cells = row.find_elements(By.TAG_NAME, "th")
                
                if cells:
                    # Extract text from cells
                    row_data = [cell.text.strip() for cell in cells]
                    
                    # Process the row data
                    try:
                        # Split player name and team if in format "Player Name - TEAM"
                        if row_data and row_data[0]:
                            player_cell = row_data[0]
                            team_code = None
                            
                            # Enhanced team extraction: check both "Name - TEAM" and other formats
                            if " - " in player_cell:
                                player_name, team_code = player_cell.split(" - ", 1)
                                row_data[0] = player_name.strip()
                                logging.info(f"Extracted team '{team_code}' from player '{player_name}'")
                                
                                # Add team code to Team column or create it
                                team_column_added = False
                                
                                if "Team" in headers:
                                    team_idx = headers.index("Team")
                                    if team_idx < len(row_data):
                                        row_data[team_idx] = team_code
                                    else:
                                        # Extend row_data to include Team
                                        row_data.extend([""] * (team_idx - len(row_data)) + [team_code])
                                    team_column_added = True
                                
                                # If no Team column exists yet, add it at the end
                                if not team_column_added and team_code:
                                    if "Team" not in headers:
                                        headers.append("Team")
                                        # Add empty team for previously processed rows
                                        for prev_row in page_data:
                                            prev_row.append("")
                                    # Add team to this row
                                    row_data.append(team_code)
                        
                        # Add or update Season column
                        if "Season" in headers:
                            season_idx = headers.index("Season")
                            if season_idx < len(row_data):
                                row_data[season_idx] = season
                            else:
                                # Extend row_data to include Season
                                row_data.extend([""] * (season_idx - len(row_data)) + [season])
                        else:
                            # Season column doesn't exist, add it
                            headers.append("Season")
                            # Add empty season for previously processed rows
                            for prev_row in page_data:
                                prev_row.append(season)
                            # Add season to this row
                            row_data.append(season)
                        
                        # Make sure row_data matches headers in length
                        if len(row_data) != len(headers):
                            if len(row_data) < len(headers):
                                # Pad with empty strings
                                row_data.extend([''] * (len(headers) - len(row_data)))
                            elif len(row_data) > len(headers):
                                # Truncate
                                row_data = row_data[:len(headers)]
                        
                        # CRITICAL FIX: Actually append the row data to page_data
                        page_data.append(row_data)
                        page_has_data = True
                    except Exception as e:
                        logging.warning(f"Error processing row: {str(e)[:100]}...")
            
            # Check if we got any data from this page
            if not page_has_data:
                logging.warning(f"No valid data processed on page {current_page}, stopping pagination")
                page_limit_reached = True
                continue
            
            logging.info(f"Processed {len(page_data)} valid rows from page {current_page}")
            all_player_data.extend(page_data)
            
            # Simple pagination: just go to the next page until we hit max pages
            # or find an empty page (which should be caught by page_has_data check above)
            if current_page < max_pages:
                current_page += 1
            else:
                logging.info(f"Reached maximum page limit ({max_pages})")
                page_limit_reached = True
        
    except Exception as e:
        logging.error(f"Error during scraping: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        # Always close the driver
        logging.info("Closing WebDriver")
        driver.quit()
    
    logging.info(f"Scraping complete. Found {len(all_player_data)} player records")
    
    # Create DataFrame from the data
    if all_player_data and headers:
        try:
            df = pd.DataFrame(all_player_data, columns=headers)
            logging.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            
            # Basic data cleaning
            non_numeric_cols = ['Player', 'Team', 'Season']
            potential_numeric_cols = [h for h in headers if h not in non_numeric_cols]
            
            for col in potential_numeric_cols:
                if col in df.columns:
                    # Replace placeholders with NaN
                    df[col] = df[col].replace(['--', '', 'N/A', 'NA', '-'], pd.NA)
                    # Handle percentage columns
                    if '%' in col or df[col].astype(str).str.contains('%').any():
                        df[col] = df[col].astype(str).str.replace('%', '').replace(',', '')
                    # Convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove duplicates
            before_drop = len(df)
            df = df.drop_duplicates()
            after_drop = len(df)
            if before_drop > after_drop:
                logging.info(f"Removed {before_drop - after_drop} duplicate entries")
            
            return df
            
        except Exception as e:
            logging.error(f"Error creating DataFrame: {str(e)}")
            # Return raw data if DataFrame creation fails
            return {'headers': headers, 'data': all_player_data}
    
    elif all_player_data:
        logging.info("Data scraped, but headers not finalized")
        return {'data': all_player_data}
    
    else:
        logging.warning("No data was scraped")
        return pd.DataFrame()

# --- Function to discover available teams ---
def discover_teams():
    """Discover all available teams on the UFA stats page"""
    teams = ["All"]  # Default option
    
    logging.info("Discovering available teams...")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Setup WebDriver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        # Visit the stats page
        driver.get(BASE_URL)
        time.sleep(INITIAL_WAIT)
        
        # Try to find the team dropdown
        try:
            team_select = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    "select#edit-team, select[name*='team']"))
            )
            
            # Get all team options
            options = team_select.find_elements(By.TAG_NAME, "option")
            for option in options:
                team_name = option.text.strip()
                if team_name and team_name != "All":
                    teams.append(team_name)
            
            logging.info(f"Discovered {len(teams)} teams: {', '.join(teams[:5])}...")
        
        except Exception as e:
            logging.warning(f"Could not discover teams: {str(e)[:100]}...")
    
    finally:
        driver.quit()
    
    return teams

# --- Main Function to Scrape All Seasons ---
def scrape_all_seasons(testing_mode=True):
    """
    Scrape data for all seasons and combine into one dataset.
    
    Args:
        testing_mode: If True, only scrape TEST_PAGES from each season
    
    Returns:
        Dictionary with information about what was scraped
    """
    # Setup logging
    log_file = setup_logging()
    
    # Create output directory
    output_dir = "scraped_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Discover available teams first
    available_teams = discover_teams()
    
    # Start with Career and then individual seasons
    all_data = []
    standard_columns = []  # Track all columns to standardize across seasons
    seasons_scraped = 0
    total_players = 0
    
    for season in SEASONS:
        logging.info(f"{'='*40}")
        logging.info(f"Scraping Season: {season}")
        logging.info(f"{'='*40}")
        
        # For each season, scrape with Total stats
        season_data = scrape_ufa_stats(
            max_pages=MAX_PAGES,
            start_page=1,
            season=season,
            per="Total",  # Always use Total stats
            team="All",
            testing_mode=testing_mode
        )
        
        # Save season data
        if isinstance(season_data, pd.DataFrame) and not season_data.empty:
            seasons_scraped += 1
            total_players += len(season_data)
            
            # Update standard columns list with any new columns found
            for col in season_data.columns:
                if col not in standard_columns:
                    standard_columns.append(col)
                    
            # Ensure Team column exists
            if 'Team' not in season_data.columns:
                season_data['Team'] = ""
                # Try to extract team info from Player column if available
                if 'Player' in season_data.columns:
                    for idx, player in enumerate(season_data['Player']):
                        if ' - ' in player:
                            player_name, team_code = player.split(' - ', 1)
                            season_data.at[idx, 'Player'] = player_name.strip()
                            season_data.at[idx, 'Team'] = team_code.strip()
                            logging.info(f"Post-processing: Extracted team '{team_code}' from player '{player_name}'")
            
            # Save season data to CSV
            season_file = os.path.join(output_dir, f"ufa_player_stats_{season}.csv")
            season_data.to_csv(season_file, index=False)
            logging.info(f"Saved {len(season_data)} rows for {season} to {season_file}")
            
            # Add to combined data
            all_data.append(season_data)
            
        elif isinstance(season_data, dict) and 'data' in season_data:
            # If we got raw data, convert to DataFrame
            headers = season_data.get('headers', [f"Col{i}" for i in range(len(season_data['data'][0]))])
            season_df = pd.DataFrame(season_data['data'], columns=headers)
            
            # Update standard columns
            for col in season_df.columns:
                if col not in standard_columns:
                    standard_columns.append(col)
            
            # Save to CSV
            season_file = os.path.join(output_dir, f"ufa_player_stats_{season}.csv")
            season_df.to_csv(season_file, index=False)
            logging.info(f"Saved {len(season_df)} rows for {season} to {season_file}")
            
            # Add to combined data
            all_data.append(season_df)
    
    # Combine all season data if available
    if all_data:
        # First standardize all DataFrames to have the same columns
        standardized_data = []
        logging.info(f"Standardizing columns across all seasons: {standard_columns}")
        
        for df in all_data:
            # Add missing columns
            for col in standard_columns:
                if col not in df.columns:
                    df[col] = ""
            
            # Reorder columns to match standard order
            df = df[standard_columns]
            standardized_data.append(df)
        
        # Combine the standardized data
        combined_df = pd.concat(standardized_data, ignore_index=True)
        combined_file = os.path.join(output_dir, "ufa_player_stats.csv")
        combined_df.to_csv(combined_file, index=False)
        logging.info(f"Saved combined dataset with {len(combined_df)} rows to {combined_file}")
    else:
        logging.warning("No data collected from any season")
    
    logging.info(f"Scraping complete. Log saved to: {log_file}")
    
    # Return information about what was scraped
    return {
        "seasons_scraped": seasons_scraped,
        "total_players": total_players,
        "columns": standard_columns
    }

# --- Run the scraper ---
if __name__ == "__main__":
    print("Starting UFA Stats Scraper...")
    scrape_all_seasons(testing_mode=False)  # Changed to scrape all pages 