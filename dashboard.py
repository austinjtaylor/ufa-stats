import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
import plotly.graph_objects as go


def main():
    st.set_page_config(page_title="UFA Player Stats Dashboard", page_icon="🏒", layout="wide")
    
    # Initialize session state for storing notes
    if "per_note" not in st.session_state:
        st.session_state["per_note"] = ""
    
    # Dark theme styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main {
        background-color: #262730;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16idsys h1 {
        color: #FAFAFA;
    }
    .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16idsys h2 {
        color: #FAFAFA;
    }
    .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16idsys h3 {
        color: #FAFAFA;
    }
    .stSlider > div > div > div > div {
        background-color: #4F8BF9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- HEADER ---
    st.title("UFA Player Stats Dashboard")
    st.markdown("Interactive dashboard for analyzing UFA player statistics")
    
    # --- DATA LOADING ---
    @st.cache_data
    def load_data(file_path='scraped_data/ufa_player_stats.csv'):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("The dataframe is empty")
            
            # Clean up the data
            if 'Player' in df.columns:
                df['Player'] = df['Player'].fillna('Unknown').astype(str)
            if 'Team' in df.columns:
                df['Team'] = df['Team'].fillna('Unknown').astype(str)
                df['Team'] = df['Team'].apply(
                    lambda x: x.split(',')[0].strip() if ',' in x else x
                )
            if 'Season' in df.columns:
                df['Season'] = df['Season'].astype(str).str.replace(
                    r'\.0$', '', regex=True
                )
            
            # Handle team name changes: AUS -> ATX from 2022 onwards
            if 'Team' in df.columns and 'Season' in df.columns:
                # Create a mask for rows where Team is AUS and Season is 2021 or before
                season_numeric = pd.to_numeric(df['Season'], errors='coerce')
                aus_mask = (df['Team'] == 'AUS') & (season_numeric <= 2021)
                # Update team name for those rows
                df.loc[aus_mask, 'Team'] = 'ATX'
            
            # Remove the Per column if it exists
            if 'Per' in df.columns:
                df = df.drop(columns=['Per'])
                
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return generate_sample_data()
    
    @st.cache_data
    def generate_sample_data(rows=100):
        """Generate sample data for demonstration when actual data isn't available"""
        np.random.seed(42)
        
        players = [f"Player {i}" for i in range(1, rows+1)]
        teams = np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], size=rows)
        seasons = np.random.choice(
            ['Career', '2023-2024', '2022-2023', '2021-2022'], size=rows
        )
        
        # Sample data with column definitions
        data = {
            'Player': players,
            'Team': teams,
            'Season': seasons,
            'G': np.random.randint(1, 82, size=rows),         # Games played
            'PP': np.random.randint(50, 500, size=rows),      # Points Played
            'POS': np.random.randint(20, 200, size=rows),     # Possessions
            'SCR': np.random.randint(0, 70, size=rows),       # Scores
            'AST': np.random.randint(0, 70, size=rows),       # Assists
            'GLS': np.random.randint(0, 50, size=rows),       # Goals
            'BLK': np.random.randint(0, 40, size=rows),       # Blocks
            '+/-': np.random.randint(-30, 30, size=rows),     # Plus/Minus
            'Cmp': np.random.randint(50, 300, size=rows),     # Completions
            'Cmp%': np.random.randint(70, 100, size=rows),    # Completion percentage
            'Y': np.random.randint(1000, 6000, size=rows),    # Total Yards
            'TY': np.random.randint(500, 3000, size=rows),    # Throwing yards
            'RY': np.random.randint(500, 5000, size=rows),    # Receiving yards
            'OEFF': np.random.uniform(40, 70, size=rows).round(2),  # Off. efficiency
            'HA': np.random.randint(5, 30, size=rows),        # Hockey Assists
            'T': np.random.randint(5, 25, size=rows),         # Throwaways
            'Hck': np.random.randint(1, 100, size=rows),      # Hucks
            'Hck%': np.random.randint(30, 90, size=rows),     # Huck percentage
        }
        return pd.DataFrame(data)
    
    # --- DATA CLEANING ---
    def clean_and_convert_data(df):
        """Clean the data and convert columns to appropriate types"""
        for col in df.columns:
            # Handle string columns
            if col in ['Player', 'Team', 'Season']:
                df[col] = df[col].fillna('Unknown').astype(str)
                if col == 'Season':
                    df[col] = df[col].str.replace(r'\.0$', '', regex=True)
                if col == 'Team':
                    df[col] = df[col].apply(
                        lambda x: x.split(',')[0].strip() if ',' in x else x
                    )
                continue
            
            try:
                # Handle numeric columns
                if df[col].dtype == 'object':
                    # Handle percentage columns
                    if '%' in col or df[col].astype(str).str.contains('%').any():
                        df[col] = df[col].astype(str).str.replace('%', '')
                    
                    # Handle commas in numbers
                    df[col] = df[col].astype(str).str.replace(',', '')
                    
                    # Replace missing values
                    df[col] = df[col].replace(['--', '', 'N/A', 'NA', '-'], pd.NA)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                continue
        
        return df
    
    # Load the data
    original_df = load_data()
    df = clean_and_convert_data(original_df.copy())
    
    # Define key column names directly instead of using detection functions
    # These are the known column names from our sample data structure
    player_col = 'Player'
    team_col = 'Team'
    season_col = 'Season'
    plusminus_col = '+/-'
    
    # Define default filter columns as requested
    default_filter_cols = ['G', 'PP', 'POS', 'SCR', 'AST', 'GLS']
    
    # Detect numeric columns for visualizations and additional filters
    def detect_numeric_columns(df):
        """Detect columns with numeric data in their original order"""
        numeric_cols = []
        for col in df.columns:
            try:
                if col in ['Player', 'Team', 'Season']:
                    continue
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                if (not numeric_vals.isna().all() and 
                        numeric_vals.notna().sum() > len(df) * 0.5):
                    numeric_cols.append(col)
            except Exception:
                continue
        return numeric_cols
    
    numeric_cols = detect_numeric_columns(df)
    
    # Make a copy for filtering
    df_filtered = df.copy()
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    # Reset Filters button
    if st.sidebar.button("Reset All Filters"):
        st.session_state.clear()
        # Ensure the season filter is reset to "Career" 
        if "selected_season" in st.session_state:
            st.session_state["selected_season"] = "Career"
        # Ensure the per filter is reset to "Total"
        if "selected_per" in st.session_state:
            st.session_state["selected_per"] = "Total"
        st.rerun()
    
    # Season filter
    selected_season = None
    if season_col in df.columns:
        seasons = sorted(
            df[season_col].astype(str).str.replace(r'\.0$', '', regex=True).unique(), 
            reverse=True
        )
        
        default_idx = seasons.index("Career") if "Career" in seasons else 0
        
        # Initialize session state for season if it doesn't exist
        if "selected_season" not in st.session_state:
            st.session_state["selected_season"] = seasons[default_idx]
            
        selected_season = st.sidebar.selectbox(
            "Season",
            options=list(seasons),
            index=default_idx,
            key="selected_season"
        )
        
        df_filtered = df[
            df[season_col].astype(str).str.replace(r'\.0$', '', regex=True) == selected_season
        ]
    
    # Team filter
    selected_team = "All"
    if team_col in df.columns:
        if selected_season == "Career":
            all_teams = sorted(df[team_col].fillna('Unknown').astype(str).unique())
            teams = [team for team in all_teams if team and team != 'Unknown']
            if not teams:
                teams = ['Unknown']
        else:
            teams = sorted(df_filtered[team_col].fillna('Unknown').astype(str).unique())
            teams = [team for team in teams if team and team != 'Unknown']
            if not teams:
                teams = ['Unknown']
            
        selected_team = st.sidebar.selectbox(
            "Team",
            options=["All"] + list(teams)
        )
        
        if selected_team != "All":
            if selected_season == "Career":
                # For Career + Team, aggregate data across all seasons for this team
                team_data = original_df[
                    original_df[team_col].fillna('Unknown').astype(str) == selected_team
                ]
                
                # Only keep data from numeric seasons (exclude existing Career entries)
                team_seasons = team_data[
                    ~(team_data[season_col].astype(str).str.lower() == "career")
                ]
                
                if not team_seasons.empty:
                    # Group by player and aggregate
                    agg_numeric_cols = [
                        col for col in team_seasons.columns 
                        if col not in [player_col, team_col, season_col]
                        and pd.api.types.is_numeric_dtype(team_seasons[col])
                    ]
                    
                    # Create aggregation dictionary
                    agg_dict = {col: 'sum' for col in agg_numeric_cols}
                    if team_col in team_seasons.columns:
                        agg_dict[team_col] = 'first'
                    if season_col in team_seasons.columns:
                        agg_dict[season_col] = 'first'
                    
                    # Group by player and compute career totals
                    career_for_team = team_seasons.groupby(
                        player_col
                    ).agg(agg_dict).reset_index(drop=False)
                    
                    career_for_team[season_col] = "Career"
                    
                    # Sort by +/- column if available
                    if plusminus_col in career_for_team.columns:
                        career_for_team = career_for_team.sort_values(
                            by=plusminus_col, ascending=False
                        )
                    
                    # Reorder columns to ensure Player is first
                    if player_col in career_for_team.columns:
                        cols = [player_col] + [
                            col for col in career_for_team.columns if col != player_col
                        ]
                        career_for_team = career_for_team[cols]
                    
                    df_filtered = career_for_team
                else:
                    df_filtered = pd.DataFrame(columns=df.columns)
            else:
                # Regular season + team filtering
                df_filtered = df_filtered[
                    df_filtered[team_col].fillna('Unknown').astype(str) == selected_team
                ]
    
    # Per filter (normalization)
    per_options = ["Total", "Per Game", "Per 10 points", "Per 10 poss.", "Per 100 min."]
    per_help = """
    Normalize statistics based on:
    - Total: Raw totals
    - Per Game: Stats per game played (G)
    - Per 10 points: Stats per 10 points played (PP)
    - Per 10 poss.: Stats per 10 possessions (POS)
    - Per 100 min.: Stats per 100 minutes played (uses MIN if available or G as fallback)
    """
    
    # Initialize session state for per if it doesn't exist
    if "selected_per" not in st.session_state:
        st.session_state["selected_per"] = "Total"
        
    selected_per = st.sidebar.selectbox(
        "Per",
        options=per_options,
        index=per_options.index(st.session_state["selected_per"]),  # Use session state value
        key="selected_per",
        help=per_help
    )
    
    # Apply the Per filter normalization
    if selected_per != "Total" and not df_filtered.empty:
        # Create a copy for normalization
        normalized_df = df_filtered.copy()
        
        # Define normalization factors and divisor columns
        if selected_per == "Per Game":
            divisor_col = 'G'  # Games played
            factor = 1
            # Apply minimum threshold (10 games for career stats)
            if selected_season == "Career":
                normalized_df = normalized_df[normalized_df[divisor_col] >= 10]
            note = "* Players must have at least 10 games played to be included in career stats."
            
        elif selected_per == "Per 10 points":
            divisor_col = 'PP'  # Points Played
            factor = 10
            # Apply minimum threshold (100 points for career, 20 for season)
            if selected_season == "Career":
                normalized_df = normalized_df[normalized_df[divisor_col] >= 100]
                # Filter out data before 2013 for career stats
                if season_col in normalized_df.columns:
                    try:
                        # Attempt to filter only if needed
                        seasons_str = normalized_df[season_col].astype(str)
                        if any(s.isdigit() and int(s) < 2013 for s in seasons_str):
                            st.sidebar.warning("⚠️ Some data before 2013 excluded due to inaccurate points data.")
                    except:
                        pass
            else:
                # For individual seasons, use minimum of 20
                normalized_df = normalized_df[normalized_df[divisor_col] >= 20]
                # Warn for seasons before 2013
                try:
                    if selected_season.isdigit() and int(selected_season) < 2013:
                        st.sidebar.warning("⚠️ Data before 2013 may have inaccurate points played information.")
                except:
                    pass
            note = ("* Must have 100 points played to be included in career stats (20 for individual seasons). "
                   "** Stats before 2013 may be inaccurate due to incomplete points data.")
                
        elif selected_per == "Per 10 poss.":
            divisor_col = 'POS'  # Possessions
            factor = 10
            # Apply minimum threshold (100 possessions for career, 20 for season)
            if selected_season == "Career":
                normalized_df = normalized_df[normalized_df[divisor_col] >= 100]
                # Filter out data before 2014 for career stats
                if season_col in normalized_df.columns:
                    try:
                        # Attempt to filter only if needed
                        seasons_str = normalized_df[season_col].astype(str)
                        if any(s.isdigit() and int(s) < 2014 for s in seasons_str):
                            st.sidebar.warning("⚠️ Some data before 2014 excluded due to inaccurate possessions data.")
                    except:
                        pass
            else:
                # For individual seasons, use minimum of 20
                normalized_df = normalized_df[normalized_df[divisor_col] >= 20]
                # Warn for seasons before 2014
                try:
                    if selected_season.isdigit() and int(selected_season) < 2014:
                        st.sidebar.warning("⚠️ Data before 2014 may have inaccurate possessions information.")
                except:
                    pass
            note = ("* Must have 100 offensive possessions to be included in career stats (20 for individual seasons). "
                   "** Stats before 2014 may be inaccurate due to incomplete possessions data.")
                
        elif selected_per == "Per 100 min.":
            # Look for a minutes column, or use 'G' as fallback
            if 'MIN' in normalized_df.columns:
                divisor_col = 'MIN'
                # Apply minimum threshold (100 minutes for career, 20 for season)
                if selected_season == "Career":
                    normalized_df = normalized_df[normalized_df[divisor_col] >= 100]
                    # Filter out data before 2014 for career stats
                    if season_col in normalized_df.columns:
                        try:
                            # Attempt to filter only if needed
                            seasons_str = normalized_df[season_col].astype(str)
                            if any(s.isdigit() and int(s) < 2014 for s in seasons_str):
                                st.sidebar.warning("⚠️ Some data before 2014 excluded due to inaccurate minutes data.")
                        except:
                            pass
                else:
                    # For individual seasons, use minimum of 20
                    normalized_df = normalized_df[normalized_df[divisor_col] >= 20]
                    # Warn for seasons before 2014
                    try:
                        if selected_season.isdigit() and int(selected_season) < 2014:
                            st.sidebar.warning("⚠️ Data before 2014 may have inaccurate minutes information.")
                    except:
                        pass
                note = ("* Must have 100 minutes played to be included in career stats (20 for individual seasons). "
                       "** Stats before 2014 may be inaccurate due to incomplete minutes data.")
            else:
                divisor_col = 'G'  # Use games as fallback if minutes not available
                # Apply minimum threshold (10 games for career stats)
                if selected_season == "Career":
                    normalized_df = normalized_df[normalized_df[divisor_col] >= 10]
                note = "* Using games played as a proxy for minutes (minutes data not available)."
            factor = 100
        else:
            note = ""
        
        # Make sure the divisor column exists and is valid
        if divisor_col in normalized_df.columns:
            try:
                # Apply normalization to numeric columns only
                numeric_columns = [
                    col for col in normalized_df.columns 
                    if col not in [player_col, team_col, season_col] 
                    and pd.api.types.is_numeric_dtype(normalized_df[col])
                ]
                
                # Don't normalize divisor columns
                if divisor_col in numeric_columns:
                    numeric_columns.remove(divisor_col)
                
                # Replace zeros in divisor with NaN to avoid division by zero
                normalized_df[divisor_col] = normalized_df[divisor_col].replace(0, np.nan)
                
                for col in numeric_columns:
                    # Skip percentage columns and efficiency metrics
                    if '%' in col or col.lower().endswith('pct') or col.lower().endswith('eff'):
                        continue
                    
                    # Normalize the column based on the selected option
                    normalized_df[col] = (normalized_df[col] / normalized_df[divisor_col]) * factor
                    
                    # Round to 2 decimal places for better readability
                    normalized_df[col] = normalized_df[col].round(2)
                
                # Update the filtered dataframe with normalized values
                df_filtered = normalized_df
                
                # Store note in session state to display at bottom of page
                if selected_per != "Total":
                    st.session_state["per_note"] = note
                else:
                    st.session_state["per_note"] = ""
                    
            except Exception as e:
                st.sidebar.warning(f"Could not apply '{selected_per}' normalization: {e}")
        else:
            st.sidebar.warning(f"Cannot apply '{selected_per}' normalization: {divisor_col} column not found")
    else:
        # Clear note if not using a Per filter
        st.session_state["per_note"] = ""
    
    # Player name filter
    if player_col in df.columns:
        player_search = st.sidebar.text_input("Search by Player Name")
        if player_search and player_col in df_filtered.columns:
            df_filtered[player_col] = df_filtered[player_col].fillna('Unknown').astype(str)
            mask = df_filtered[player_col].str.contains(
                player_search, case=False, na=False
            )
            df_filtered = df_filtered[mask]
    
    # --- NUMERIC FILTERS ---
    st.sidebar.subheader("Filter by:")
    
    # Store filters for display
    default_filters = {}
    all_filters = {}
    
    # Set up default filters for the first 6 columns as requested
    for col in default_filter_cols:
        if col in df_filtered.columns:
            try:
                col_values = df_filtered[col].dropna()
                
                if len(col_values) > 0:
                    min_val = int(col_values.min()) if not pd.isna(col_values.min()) else 0
                    max_val = int(col_values.max()) if not pd.isna(col_values.max()) else 100
                else:
                    min_val = 0
                    max_val = 100
                
                # Ensure min and max are not equal
                if min_val == max_val:
                    max_val = min_val + 1
                    
                default_filters[col] = {
                    "label": f"{col}",
                    "min": min_val,
                    "max": max_val,
                    "default": (min_val, max_val)
                }
            except Exception as e:
                st.warning(f"Could not add filter for {col}: {e}")
    
    # Add remaining numeric columns as optional filters
    for col in numeric_cols:
        if col not in default_filters:
            try:
                col_min = df_filtered[col].min()
                col_max = df_filtered[col].max()
                
                # Ensure values are numeric
                if pd.isna(col_min) or not isinstance(col_min, (int, float)):
                    min_val = 0
                else:
                    min_val = int(col_min)
                    
                if pd.isna(col_max) or not isinstance(col_max, (int, float)):
                    max_val = 100
                else:
                    max_val = int(col_max)
                
                # Ensure min and max are not equal
                if min_val == max_val:
                    max_val = min_val + 1
                    
                all_filters[col] = {
                    "label": f"{col}",
                    "min": min_val,
                    "max": max_val,
                    "default": (min_val, max_val)
                }
            except Exception:
                continue
    
    # Make sure Hck% is available in the filters if it exists in the dataframe
    if 'Hck%' in df_filtered.columns and 'Hck%' not in all_filters and 'Hck%' not in default_filters:
        try:
            hck_pct_min = int(df_filtered['Hck%'].min()) if not pd.isna(df_filtered['Hck%'].min()) else 0
            hck_pct_max = int(df_filtered['Hck%'].max()) if not pd.isna(df_filtered['Hck%'].max()) else 100
            
            # Ensure min and max are not equal
            if hck_pct_min == hck_pct_max:
                hck_pct_max = hck_pct_min + 1
                
            all_filters['Hck%'] = {
                "label": "Hck%",
                "min": hck_pct_min,
                "max": hck_pct_max,
                "default": (hck_pct_min, hck_pct_max)
            }
        except Exception:
            pass
    
    # Apply default filters
    for col, filter_info in default_filters.items():
        filter_range = st.sidebar.slider(
            filter_info["label"],
            min_value=filter_info["min"],
            max_value=filter_info["max"],
            value=filter_info["default"]
        )
        df_filtered = df_filtered[
            (df_filtered[col] >= filter_range[0]) & 
            (df_filtered[col] <= filter_range[1])
        ]
    
    # Additional filters option
    if all_filters:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Additional Filters")
        
        # Reorder the filter options to put Hck% after Hck and before PUL
        filter_options = list(all_filters.keys())
        
        # Find indices for special ordering
        hck_index = None
        pul_index = None
        
        # First pass to find indices
        for i, col in enumerate(filter_options):
            if col.upper() == 'HCK':
                hck_index = i
            elif col.upper() == 'PUL':
                pul_index = i
        
        # Handle special reordering for Hck%
        if 'Hck%' in filter_options:
            # Remove Hck% first to avoid duplication
            filter_options.remove('Hck%')
            
            # Place Hck% after Hck and before PUL if possible
            if hck_index is not None:
                filter_options.insert(hck_index + 1, 'Hck%')
            elif pul_index is not None:
                filter_options.insert(pul_index, 'Hck%')
            else:
                # If neither Hck nor PUL exists, just add to the end
                filter_options.append('Hck%')
        
        selected_additional_filters = st.sidebar.multiselect(
            "Select additional metrics to filter by:",
            options=filter_options
        )
        
        # Apply selected additional filters
        for col in selected_additional_filters:
            filter_info = all_filters[col]
            filter_range = st.sidebar.slider(
                filter_info["label"],
                min_value=filter_info["min"],
                max_value=filter_info["max"],
                value=filter_info["default"]
            )
            df_filtered = df_filtered[
                (df_filtered[col] >= filter_range[0]) & 
                (df_filtered[col] <= filter_range[1])
            ]
    
    # --- DISPLAY DATA ---
    st.header("Filtered Player Stats")
    
    # Prepare display dataframe
    display_df = df_filtered.copy()
    
    # Sort by +/- descending by default if it exists
    if plusminus_col in display_df.columns:
        display_df = display_df.sort_values(by=plusminus_col, ascending=False)
    
    # Hide Season and Team columns as they're used as filters
    columns_to_hide = []
    if season_col in display_df.columns:
        columns_to_hide.append(season_col)
    if team_col in display_df.columns:
        columns_to_hide.append(team_col)
    
    if columns_to_hide:
        display_df = display_df.drop(columns=columns_to_hide)
    
    # Reset index for clean numbered rows
    display_df = display_df.reset_index(drop=True)
    
    # Show results
    if len(display_df) > 0:
        # Update header to include the "Per" selection
        if selected_per != "Total":
            st.write(f"Displaying {len(display_df)} players — {selected_per}")
        else:
            st.write(f"Displaying {len(display_df)} players")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No players match the selected filters.")
    
    # Use a color palette with good distinction
    custom_palette = px.colors.sequential.Viridis
    
    # Define a qualitative color palette for categorical coloring
    qualitative_palette = px.colors.qualitative.Bold
    
    # Define a high-contrast sequential palette for numerical features
    # Use Turbo instead of Inferno for better distinction between colors
    numerical_palette = px.colors.sequential.Turbo
    
    # --- VISUALIZATION ---
    if len(df_filtered) > 0 and len(numeric_cols) > 0:
        st.header("Visualization")

        if not df_filtered.empty:
            # Get numeric columns for visualization
            vis_numeric_cols = []
            for col in df_filtered.columns:
                try:
                    if (pd.api.types.is_numeric_dtype(df_filtered[col]) and 
                            col not in ['index']):
                        vis_numeric_cols.append(col)
                except Exception:
                    continue
            
            # Get all columns and reorder special ones
            all_metric_cols = []
            hck_index = None
            pul_index = None
            
            # First collect all columns and note positions of special columns
            for i, col in enumerate(vis_numeric_cols):
                all_metric_cols.append(col)
                if col.upper() == 'HCK':
                    hck_index = i
                elif col.upper() == 'PUL':
                    pul_index = i
            
            # Handle special reordering for Hck%
            if 'Hck%' in df_filtered.columns:
                # Remove Hck% if it's already in the list to avoid duplication
                if 'Hck%' in all_metric_cols:
                    all_metric_cols.remove('Hck%')
                
                # Place Hck% after Hck and before PUL if possible
                if hck_index is not None:
                    all_metric_cols.insert(hck_index + 1, 'Hck%')
                elif pul_index is not None:
                    all_metric_cols.insert(pul_index, 'Hck%')
                else:
                    # If neither Hck nor PUL exists, just add to the end
                    all_metric_cols.append('Hck%')
            
            # Set available metrics for visualization using the reordered list
            available_metrics = all_metric_cols
            
            # Visualization type selector
            vis_type = st.selectbox(
                "Visualization Type",
                options=["Bar Chart", "Scatter Plot", "Histogram", "Radar Chart", "Performance Quadrants"]
            )
            
            # Visualization options
            with st.expander("Visualization Options", expanded=True):
                if vis_type == "Bar Chart":
                    x_axis = st.selectbox(
                        "Select Dimension",
                        options=available_metrics,
                        index=0 if available_metrics and len(available_metrics) > 0 else None
                    )
                elif vis_type == "Scatter Plot":
                    x_axis = st.selectbox(
                        "Select X Axis",
                        options=available_metrics,
                        index=0 if available_metrics and len(available_metrics) > 0 else None
                    )
                    y_axis = st.selectbox(
                        "Select Y Axis",
                        options=available_metrics,
                        index=(1 if len(available_metrics) > 1 else 0) 
                            if available_metrics else None
                    )
                    
                    # Option to color by column
                    use_color = st.checkbox("Color points by a column", value=False)
                    color_col = None
                    if use_color:
                        all_cols = list(df_filtered.columns)
                        color_col = st.selectbox(
                            "Select column for coloring",
                            options=all_cols,
                            index=all_cols.index(player_col) if player_col in all_cols else 0
                        )
                elif vis_type == "Histogram":
                    x_axis = st.selectbox(
                        "Select Dimension for Histogram",
                        options=available_metrics,
                        index=0 if available_metrics and len(available_metrics) > 0 else None
                    )
                    
                    # Number of bins option
                    n_bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
                elif vis_type == "Radar Chart":
                    # Allow selection of up to 5 players for comparison
                    if player_col in df_filtered.columns:
                        players_list = sorted(df_filtered[player_col].unique())
                        selected_players = st.multiselect(
                            "Select Players to Compare (max 5)", 
                            options=players_list,
                            max_selections=5
                        )
                        
                        # Select metrics for radar chart (3-8)
                        selected_metrics = st.multiselect(
                            "Select Metrics for Comparison (3-8 recommended)",
                            options=available_metrics,
                            default=available_metrics[:min(5, len(available_metrics))]
                        )
                    else:
                        st.warning("Player column needed for radar chart visualization.")
                elif vis_type == "Performance Quadrants":
                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox(
                            "X-Axis Metric",
                            options=available_metrics,
                            index=0 if available_metrics and len(available_metrics) > 0 else None
                        )
                        # Allow custom x-axis label for quadrant
                        x_quadrant_label = st.text_input("X-Axis Quadrant Label", value="Higher →")
                    
                    with col2:
                        y_axis = st.selectbox(
                            "Y-Axis Metric",
                            options=available_metrics,
                            index=1 if len(available_metrics) > 1 else 0 if available_metrics else None
                        )
                        # Allow custom y-axis label for quadrant
                        y_quadrant_label = st.text_input("Y-Axis Quadrant Label", value="Higher ↑")
                    
                    # Option to limit the number of players shown
                    show_top_n = st.slider("Number of players to show", min_value=5, max_value=50, value=20)
                    
                    # Option to select a team color
                    if team_col in df_filtered.columns:
                        use_team_color = st.checkbox("Color by team", value=True)
                    else:
                        use_team_color = False
            
            # Create visualizations
            if not df_filtered.empty:
                if vis_type == "Bar Chart" and x_axis:
                    if player_col in df_filtered.columns:
                        # Sort by selected dimension
                        df_vis = df_filtered.sort_values(by=x_axis, ascending=False).head(15)
                        df_vis[player_col] = df_vis[player_col].fillna('Unknown').astype(str)
                        fig = px.bar(
                            df_vis,
                            x=player_col,
                            y=x_axis,
                            title=f"Top 15 Players by {x_axis}" + (f" ({selected_per})" if selected_per != "Total" else ""),
                            labels={x_axis: x_axis, player_col: "Player"},
                            color_discrete_sequence=custom_palette
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Player column needed for bar chart visualization.")
                
                elif vis_type == "Scatter Plot" and x_axis and y_axis:
                    if player_col in df_filtered.columns:
                        df_filtered[player_col] = df_filtered[player_col].fillna('Unknown').astype(str)
                    
                    # Determine color palette based on selected color column type
                    if color_col:
                        is_numeric = pd.api.types.is_numeric_dtype(df_filtered[color_col])
                        
                        # Create the scatter plot with appropriate color scheme
                        fig = px.scatter(
                            df_filtered,
                            x=x_axis,
                            y=y_axis,
                            title=f"{y_axis} vs {x_axis}" + (f" ({selected_per})" if selected_per != "Total" else ""),
                            color=color_col,
                            hover_name=player_col if player_col in df_filtered.columns else None,
                            color_discrete_sequence=qualitative_palette if not is_numeric else None,
                            color_continuous_scale=numerical_palette if is_numeric else None,
                            size_max=10
                        )
                    else:
                        # Create basic scatter plot without coloring
                        fig = px.scatter(
                            df_filtered,
                            x=x_axis,
                            y=y_axis,
                            title=f"{y_axis} vs {x_axis}" + (f" ({selected_per})" if selected_per != "Total" else ""),
                            hover_name=player_col if player_col in df_filtered.columns else None,
                            color_discrete_sequence=[custom_palette[4]],
                            size_max=10
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif vis_type == "Histogram" and x_axis:
                    fig = px.histogram(
                        df_filtered,
                        x=x_axis,
                        title=f"Distribution of {x_axis}" + (f" ({selected_per})" if selected_per != "Total" else ""),
                        nbins=n_bins,
                        color_discrete_sequence=[custom_palette[4]]
                    )
                    # Add mean line
                    mean_value = df_filtered[x_axis].mean()
                    fig.add_vline(
                        x=mean_value, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Mean: {mean_value:.2f}",
                        annotation_position="top right"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Understanding the Histogram:**
                    - Each bar represents the number of players within a range of the selected metric
                    - Taller bars indicate more players in that range
                    - The red dashed line shows the mean (average) value
                    - You can adjust the number of bins to change the histogram granularity
                    """)
                
                elif vis_type == "Radar Chart" and player_col in df_filtered.columns:
                    if selected_players and selected_metrics and len(selected_metrics) >= 2:
                        # Filter dataframe to include only selected players
                        radar_df = df_filtered[df_filtered[player_col].isin(selected_players)]
                        
                        if not radar_df.empty:
                            # Create a figure with a polar subplot
                            fig = go.Figure()
                            
                            # Normalize data for better visualization (0-1 scale)
                            for metric in selected_metrics:
                                max_val = df_filtered[metric].max()
                                min_val = df_filtered[metric].min()
                                if max_val != min_val:  # Avoid division by zero
                                    radar_df[f"{metric}_norm"] = (radar_df[metric] - min_val) / (max_val - min_val)
                                else:
                                    radar_df[f"{metric}_norm"] = 1  # If all values are the same
                            
                            # Add traces for each player
                            for i, player in enumerate(selected_players):
                                player_data = radar_df[radar_df[player_col] == player]
                                if not player_data.empty:
                                    # Create values list, adding the first value again at the end to close the polygon
                                    values = [player_data[f"{metric}_norm"].iloc[0] for metric in selected_metrics]
                                    values.append(values[0])
                                    
                                    # Create labels list, with the first label repeated at the end
                                    theta = selected_metrics + [selected_metrics[0]]
                                    
                                    # Add trace
                                    fig.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=theta,
                                        fill='toself',
                                        name=player,
                                        line_color=qualitative_palette[i % len(qualitative_palette)]
                                    ))
                            
                            # Update layout
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                title=f"Player Comparison" + (f" ({selected_per})" if selected_per != "Total" else ""),
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation
                            st.markdown("""
                            **Understanding the Radar Chart:**
                            - Each axis represents a different metric (normalized to the same scale)
                            - The further from the center, the higher the value
                            - Different colors represent different players
                            - This chart shows relative strengths across multiple metrics at once
                            """)
                            
                            # Display actual values in a table
                            with st.expander("View Raw Values", expanded=False):
                                st.dataframe(
                                    radar_df[([player_col] + selected_metrics)],
                                    use_container_width=True
                                )
                        else:
                            st.warning("No data available for selected players.")
                    else:
                        st.warning("Please select at least one player and two metrics for radar chart.")
                elif vis_type == "Performance Quadrants":
                    if x_axis and y_axis and player_col in df_filtered.columns:
                        # Create a copy of the filtered dataframe with only the needed columns
                        quad_df = df_filtered[[player_col, x_axis, y_axis]].copy()
                        
                        # Check if we have team column for coloring
                        if team_col in df_filtered.columns and use_team_color:
                            quad_df[team_col] = df_filtered[team_col]
                        
                        # Drop rows with missing values
                        quad_df = quad_df.dropna(subset=[x_axis, y_axis])
                        
                        if not quad_df.empty:
                            # Calculate median values for quadrant lines
                            x_median = quad_df[x_axis].median()
                            y_median = quad_df[y_axis].median()
                            
                            # Sort by combined rank of both metrics to get top performers
                            quad_df['x_rank'] = quad_df[x_axis].rank(ascending=False)
                            quad_df['y_rank'] = quad_df[y_axis].rank(ascending=False)
                            quad_df['combined_rank'] = quad_df['x_rank'] + quad_df['y_rank']
                            quad_df = quad_df.sort_values('combined_rank').head(show_top_n)
                            
                            # Create the scatter plot
                            if team_col in quad_df.columns and use_team_color:
                                fig = px.scatter(
                                    quad_df, 
                                    x=x_axis, 
                                    y=y_axis,
                                    color=team_col,
                                    text=player_col,
                                    title=f"Performance Quadrants: {x_axis} vs {y_axis}" + 
                                          (f" ({selected_per})" if selected_per != "Total" else ""),
                                    color_discrete_sequence=qualitative_palette,
                                    height=700
                                )
                            else:
                                fig = px.scatter(
                                    quad_df, 
                                    x=x_axis, 
                                    y=y_axis,
                                    text=player_col,
                                    title=f"Performance Quadrants: {x_axis} vs {y_axis}" + 
                                          (f" ({selected_per})" if selected_per != "Total" else ""),
                                    color_discrete_sequence=[custom_palette[4]],
                                    height=700
                                )
                            
                            # Add quadrant lines
                            fig.add_vline(x=x_median, line_width=1, line_dash="dash", line_color="white")
                            fig.add_hline(y=y_median, line_width=1, line_dash="dash", line_color="white")
                            
                            # Add quadrant labels
                            x_range = quad_df[x_axis].max() - quad_df[x_axis].min()
                            y_range = quad_df[y_axis].max() - quad_df[y_axis].min()
                            
                            # Add quadrant annotations
                            fig.add_annotation(
                                x=x_median - (x_range * 0.25),
                                y=y_median + (y_range * 0.25),
                                text=f"Lower {x_axis}<br>Higher {y_axis}",
                                showarrow=False,
                                font=dict(color="rgba(255,255,255,0.5)")
                            )
                            fig.add_annotation(
                                x=x_median + (x_range * 0.25),
                                y=y_median + (y_range * 0.25),
                                text=f"Higher {x_axis}<br>Higher {y_axis}",
                                showarrow=False,
                                font=dict(color="rgba(255,255,255,0.5)")
                            )
                            fig.add_annotation(
                                x=x_median - (x_range * 0.25),
                                y=y_median - (y_range * 0.25),
                                text=f"Lower {x_axis}<br>Lower {y_axis}",
                                showarrow=False,
                                font=dict(color="rgba(255,255,255,0.5)")
                            )
                            fig.add_annotation(
                                x=x_median + (x_range * 0.25),
                                y=y_median - (y_range * 0.25),
                                text=f"Higher {x_axis}<br>Lower {y_axis}",
                                showarrow=False,
                                font=dict(color="rgba(255,255,255,0.5)")
                            )
                            
                            # Add custom axis descriptions
                            fig.add_annotation(
                                x=quad_df[x_axis].max(),
                                y=y_median,
                                text=x_quadrant_label,
                                showarrow=False,
                                font=dict(color="white")
                            )
                            fig.add_annotation(
                                x=x_median,
                                y=quad_df[y_axis].max(),
                                text=y_quadrant_label,
                                showarrow=False,
                                font=dict(color="white")
                            )
                            
                            # Configure hover text to show player names
                            fig.update_traces(
                                hoverinfo="text",
                                hovertext=quad_df[player_col],
                                marker=dict(size=12, opacity=0.8),
                                textposition="top center"
                            )
                            
                            # Update layout for better readability
                            fig.update_layout(
                                xaxis_title=x_axis,
                                yaxis_title=y_axis,
                                plot_bgcolor="rgba(0,0,0,0.1)",
                                showlegend=True if (team_col in quad_df.columns and use_team_color) else False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation
                            st.markdown(f"""
                            **Understanding the Performance Quadrants:**
                            - The chart divides players into four quadrants based on {x_axis} and {y_axis}
                            - Top-right: High in both metrics
                            - Top-left: Lower {x_axis}, higher {y_axis}
                            - Bottom-right: Higher {x_axis}, lower {y_axis}
                            - Bottom-left: Lower in both metrics
                            - Lines represent the median values
                            """)
                            
                            # Display actual values in a table
                            with st.expander("View Raw Values", expanded=False):
                                st.dataframe(
                                    quad_df[[player_col, x_axis, y_axis]],
                                    use_container_width=True
                                )
                        else:
                            st.warning("No data available with selected metrics.")
                    else:
                        st.warning("Please select two metrics for performance quadrants visualization.")
                else:
                    st.warning("Please select dimensions for visualization.")
            else:
                st.warning("No data available for visualization with current filters.")
            
            # Show message if no Hck columns found
            if not any(re.search('hck|huck', col.lower()) for col in df_filtered.columns):
                st.info("Note: Huck data (Hck, Hck%) is not available in the current dataset.")
    
    # --- STATS SUMMARY ---
    if not df_filtered.empty and numeric_cols:
        with st.expander("Summary Statistics", expanded=False):
            # Reorder columns to ensure Hck, Hck%, and PUL are in the correct order
            ordered_cols = []
            
            # First, collect columns in their original order
            for col in numeric_cols:
                # Add all columns except Hck% (which we'll reposition)
                if col.upper() != 'HCK%':
                    ordered_cols.append(col)
                    
                    # When we see Hck, add Hck% right after it
                    if col.upper() == 'HCK' and 'Hck%' in df_filtered.columns:
                        ordered_cols.append('Hck%')
            
            # If there's no Hck column but Hck% exists, add it to the end
            if 'Hck%' in df_filtered.columns and 'Hck%' not in ordered_cols:
                ordered_cols.append('Hck%')
                
            # Only keep columns that actually exist in the dataframe
            ordered_cols = [col for col in ordered_cols if col in df_filtered.columns]
            
            # Display the stats with the ordered columns
            st.write(df_filtered[ordered_cols].describe())
    
    # Footer
    st.markdown("---")
    
    # Display Per filter notes if applicable
    if "per_note" in st.session_state and st.session_state["per_note"]:
        st.markdown(f"""
        <div style="background-color: rgba(100, 100, 100, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h4>Notes on {selected_per} Filter</h4>
        <p>{st.session_state["per_note"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("Dashboard created for UFA Player Statistics Analysis")

    # --- PLAYER PROFILE CARDS ---
    if player_col in df.columns and not df_filtered.empty:
        st.header("Player Profile Cards")
        
        # Player selector
        available_players = sorted(df_filtered[player_col].unique())
        if available_players:
            selected_player_profiles = st.multiselect(
                "Select Players to View", 
                options=available_players,
                max_selections=4,  # Limit to 4 players for better layout
                help="Select up to 4 players to view their profile cards"
            )
            
            if selected_player_profiles:
                # Create a grid layout - either 1 row of up to 4 cards, or a 2x2 grid
                if len(selected_player_profiles) <= 2:
                    player_cols = st.columns(len(selected_player_profiles))
                else:
                    player_cols = st.columns(2)  # 2 columns for 3-4 players
                    player_cols_row2 = st.columns(2)  # Second row
                
                # Define key metrics to show on each card
                key_metrics = []
                # Add scoring metrics if available
                if 'SCR' in df_filtered.columns:
                    key_metrics.append('SCR')
                if 'GLS' in df_filtered.columns:
                    key_metrics.append('GLS')
                if 'AST' in df_filtered.columns:
                    key_metrics.append('AST')
                # Add efficiency metrics
                if 'OEFF' in df_filtered.columns:
                    key_metrics.append('OEFF')
                if 'Cmp%' in df_filtered.columns:
                    key_metrics.append('Cmp%')
                if 'Hck%' in df_filtered.columns:
                    key_metrics.append('Hck%')
                # Add volume metrics
                if 'PP' in df_filtered.columns:
                    key_metrics.append('PP')
                if 'POS' in df_filtered.columns:
                    key_metrics.append('POS')
                if '+/-' in df_filtered.columns:
                    key_metrics.append('+/-')
                # If we have fewer than 6 metrics, add more
                remaining_metrics = [col for col in numeric_cols if col not in key_metrics]
                while len(key_metrics) < 6 and remaining_metrics:
                    key_metrics.append(remaining_metrics.pop(0))
                # Limit to 6 metrics for clean display
                key_metrics = key_metrics[:6]
                
                # Display cards
                for i, player in enumerate(selected_player_profiles):
                    # Get the appropriate column based on position
                    if len(selected_player_profiles) <= 2:
                        col = player_cols[i]
                    else:
                        col = player_cols[i] if i < 2 else player_cols_row2[i-2]
                    
                    # Get player data
                    player_data = df_filtered[df_filtered[player_col] == player]
                    
                    if not player_data.empty:
                        # Create the player card
                        with col:
                            # Container with border
                            st.markdown(f"""
                            <div style="
                                border: 1px solid #4F8BF9;
                                border-radius: 10px;
                                padding: 15px;
                                margin-bottom: 20px;
                                background-color: rgba(79, 139, 249, 0.1);
                            ">
                                <h3 style="text-align: center; border-bottom: 1px solid #4F8BF9; padding-bottom: 10px;">
                                    {player}
                                </h3>
                            """, unsafe_allow_html=True)
                            
                            # Team info if available
                            if team_col in player_data.columns:
                                team = player_data[team_col].iloc[0]
                                st.markdown(f"""
                                <p style="text-align: center; margin-top: -10px; color: #8B9DC3;">
                                    {team}
                                </p>
                                """, unsafe_allow_html=True)
                            
                            # Create two columns for the metrics
                            metric_cols = st.columns(2)
                            
                            # Display metrics
                            for j, metric in enumerate(key_metrics):
                                # Check if metric exists in player data
                                if metric in player_data.columns:
                                    # Determine which column to use
                                    metric_col = metric_cols[j % 2]
                                    
                                    # Get the value
                                    value = player_data[metric].iloc[0]
                                    
                                    # Format the value based on type
                                    if metric.endswith('%') or 'EFF' in metric.upper():
                                        # Format as percentage or efficiency
                                        value_str = f"{value:.1f}"
                                    elif metric == '+/-':
                                        # Format with sign
                                        value_str = f"{value:+.1f}"
                                    else:
                                        # Regular number format
                                        value_str = f"{value:.1f}"
                                    
                                    # Display the metric
                                    with metric_col:
                                        st.markdown(f"""
                                        <div style="text-align: center; margin: 10px 0;">
                                            <p style="font-size: 0.9em; margin-bottom: 0; color: #DDDDDD;">
                                                {metric}
                                            </p>
                                            <p style="font-size: 1.2em; font-weight: bold; margin-top: 0;">
                                                {value_str}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Close the container div
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Add a small chart below the card - top 3 metrics trend
                            if selected_season == "Career" and season_col in df.columns:
                                # Get player data across seasons for this player
                                player_seasons = df[
                                    (df[player_col] == player) & 
                                    (df[season_col] != "Career")
                                ].sort_values(by=season_col)
                                
                                if len(player_seasons) > 1 and len(key_metrics) > 0:
                                    # Select top 3 metrics
                                    trend_metrics = key_metrics[:3]
                                    
                                    # Create a line chart for trends
                                    trend_data = pd.melt(
                                        player_seasons[[season_col] + trend_metrics],
                                        id_vars=[season_col],
                                        value_vars=trend_metrics,
                                        var_name='Metric',
                                        value_name='Value'
                                    )
                                    
                                    fig = px.line(
                                        trend_data,
                                        x=season_col,
                                        y='Value',
                                        color='Metric',
                                        title=f"{player} - Trends",
                                        color_discrete_sequence=qualitative_palette,
                                        height=200
                                    )
                                    fig.update_layout(
                                        margin=dict(l=10, r=10, t=30, b=10),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                        xaxis_title=None
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select one or more players to view their profile cards")

    # --- TEAM COMPARISON ---
    # This section has been commented out since it currently only sums individual player stats,
    # which is not meaningful for team comparison. Team-specific stats would be needed for 
    # a proper team comparison feature.
    # 
    # if team_col in df.columns and not df_filtered.empty and selected_season:
    #     st.header("Team Comparison")
    #     
    #     # Check if we need to recalculate team aggregates
    #     if selected_season == "Career":
    #         # For Career, we need to aggregate data by team first
    #         team_data = df[df[season_col] != "Career"].copy()
    #     else:
    #         # For specific season, filter by the season first
    #         team_data = df[df[season_col] == selected_season].copy()
    #     
    #     # Only proceed if we have team data
    #     if not team_data.empty:
    #         # Aggregate statistics by team
    #         numeric_cols_for_teams = [col for col in numeric_cols 
    #                                   if col in team_data.columns and not col.endswith('%')]
    #         
    #         # Create a list of aggregate functions for each column
    #         agg_dict = {}
    #         for col in numeric_cols_for_teams:
    #             if col in ['G', 'PP', 'POS']:  # These are counting stats that should be averaged
    #                 agg_dict[col] = 'mean'
    #             else:
    #                 agg_dict[col] = 'sum'
    #         
    #         # Add player count
    #         agg_dict[player_col] = 'count'
    #         
    #         # Aggregate data
    #         team_stats = team_data.groupby(team_col).agg(agg_dict).reset_index()
    #         team_stats = team_stats.rename(columns={player_col: 'Players'})
    #         
    #         # Calculate derived metrics
    #         if 'SCR' in team_stats.columns and 'POS' in team_stats.columns:
    #             team_stats['OEFF'] = (team_stats['SCR'] / team_stats['POS'] * 100).round(1)
    #         
    #         # Show only teams with minimum players
    #         min_players = 5
    #         team_stats = team_stats[team_stats['Players'] >= min_players]
    #         
    #         # Let user select metrics to compare
    #         if not team_stats.empty:
    #             # Visualization options
    #             st.subheader("Visualize Team Comparison")
    #             
    #             # Select metrics to visualize
    #             default_team_metrics = ['OEFF', 'SCR', '+/-'] if all(m in team_stats.columns for m in ['OEFF', 'SCR', '+/-']) else []
    #             if not default_team_metrics and len(numeric_cols_for_teams) > 0:
    #                 default_team_metrics = [numeric_cols_for_teams[0]]
    #                 
    #             selected_team_metrics = st.multiselect(
    #                 "Select metrics to compare",
    #                 options=[col for col in team_stats.columns if col not in [team_col, 'Players']],
    #                 default=default_team_metrics[:1] if default_team_metrics else [],
    #                 max_selections=3
    #             )
    #             
    #             # Select teams to compare
    #             all_teams = sorted(team_stats[team_col].unique())
    #             selected_teams = st.multiselect(
    #                 "Select teams to compare",
    #                 options=all_teams,
    #                 default=all_teams[:min(5, len(all_teams))],
    #                 max_selections=10
    #             )
    #             
    #             # Visualization type
    #             team_chart_type = st.radio(
    #                 "Chart type",
    #                 options=["Bar Chart", "Radar Chart"],
    #                 horizontal=True
    #             )
    #             
    #             # Create visualization
    #             if selected_team_metrics and selected_teams:
    #                 # Filter data
    #                 vis_data = team_stats[team_stats[team_col].isin(selected_teams)]
    #                 
    #                 if team_chart_type == "Bar Chart":
    #                     # Create grouped bar chart
    #                     team_compare_data = pd.melt(
    #                         vis_data,
    #                         id_vars=team_col,
    #                         value_vars=selected_team_metrics,
    #                         var_name='Metric',
    #                         value_name='Value'
    #                     )
    #                     
    #                     fig = px.bar(
    #                         team_compare_data,
    #                         x=team_col,
    #                         y='Value',
    #                         color='Metric',
    #                         barmode='group',
    #                         title=f"Team Comparison - {selected_season}",
    #                         color_discrete_sequence=qualitative_palette,
    #                         height=500
    #                     )
    #                     
    #                     # Update layout
    #                     fig.update_layout(
    #                         xaxis_title="Team",
    #                         yaxis_title="Value",
    #                         legend_title="Metric",
    #                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    #                     )
    #                     
    #                     st.plotly_chart(fig, use_container_width=True)
    #                     
    #                 else:  # Radar Chart
    #                     # Create a normalized version of the data for radar
    #                     radar_data = vis_data.copy()
    #                     
    #                     # Normalize each metric between 0 and 1
    #                     for metric in selected_team_metrics:
    #                         max_val = radar_data[metric].max()
    #                         min_val = radar_data[metric].min()
    #                         if max_val != min_val:
    #                             radar_data[f"{metric}_norm"] = (radar_data[metric] - min_val) / (max_val - min_val)
    #                         else:
    #                             radar_data[f"{metric}_norm"] = 1
    #                     
    #                     # Create radar chart
    #                     fig = go.Figure()
    #                     
    #                     for i, team in enumerate(selected_teams):
    #                         team_row = radar_data[radar_data[team_col] == team]
    #                         
    #                         if not team_row.empty:
    #                             # Extract normalized values
    #                             values = [team_row[f"{metric}_norm"].iloc[0] for metric in selected_team_metrics]
    #                             # Repeat first value to close the polygon
    #                             values.append(values[0])
    #                             
    #                             # Labels, also repeating first to close the loop
    #                             theta = selected_team_metrics + [selected_team_metrics[0]]
    #                             
    #                             # Add trace
    #                             fig.add_trace(go.Scatterpolar(
    #                                 r=values,
    #                                 theta=theta,
    #                                 fill='toself',
    #                                 name=team,
    #                                 line_color=qualitative_palette[i % len(qualitative_palette)]
    #                             ))
    #                     
    #                     # Update layout
    #                     fig.update_layout(
    #                         polar=dict(
    #                             radialaxis=dict(
    #                                 visible=True,
    #                                 range=[0, 1]
    #                             )
    #                         ),
    #                         title=f"Team Comparison - {selected_season}",
    #                         showlegend=True,
    #                         height=600
    #                     )
    #                     
    #                     st.plotly_chart(fig, use_container_width=True)
    #                     
    #                     # Display actual values in a table
    #                     with st.expander("View Raw Team Values"):
    #                         st.dataframe(
    #                             vis_data[[team_col] + selected_team_metrics],
    #                             use_container_width=True
    #                         )
    #                 # else:
    #                 #    st.info("Please select metrics and teams to compare")
    #             else:
    #                 st.info("Please select metrics and teams to compare")
    #         else:
    #             st.info(f"Not enough teams with {min_players}+ players in the selected season")


def is_string_column(column):
    """Check if a column is likely to contain string data (player names)"""
    is_object = column.dtype == 'object'
    has_strings = column.astype(str).str.isalpha().any()
    return is_object and has_strings


if __name__ == "__main__":
    main() 