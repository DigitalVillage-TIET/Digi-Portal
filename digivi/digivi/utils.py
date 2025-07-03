import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import logging
from django.conf import settings
from io import BytesIO

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="info"):
    """Print colored status messages with icons"""
    if status == "success":
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")
    elif status == "error":
        print(f"{Colors.RED}✗ {message}{Colors.END}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
    elif status == "info":
        print(f"{Colors.BLUE}ℹ {message}{Colors.END}")
    elif status == "process":
        print(f"{Colors.CYAN}⚡ {message}{Colors.END}")

def get_google_sheets_client():
    """
    Create and return a Google Sheets client using service account credentials.
    
    Returns:
        gspread.Client: Authenticated Google Sheets client
    """
    try:
        print_status("Initializing Google Sheets client...", "process")
        
        # Define the scope for Google Sheets API
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        print_status("API scopes configured", "success")
        
        # Load credentials from service account key file
        # Make sure to set GOOGLE_SERVICE_ACCOUNT_KEY_PATH in your settings
        print_status("Loading service account credentials...", "process")
        credentials = Credentials.from_service_account_file(
            settings.GOOGLE_SERVICE_ACCOUNT_KEY_PATH, 
            scopes=scope
        )
        print_status("Service account credentials loaded", "success")
        
        # Create and return the client
        print_status("Authorizing Google Sheets client...", "process")
        client = gspread.authorize(credentials)
        print_status("Google Sheets client authorized successfully", "success")
        return client
    
    except Exception as e:
        print_status(f"Failed to create Google Sheets client: {e}", "error")
        logger.error(f"Error creating Google Sheets client: {e}")
        return None

def get_google_sheets_client_from_dict():
    """
    Alternative method: Create Google Sheets client from credentials dictionary.
    Use this if you store credentials as a dictionary in settings instead of a file.
    
    Returns:
        gspread.Client: Authenticated Google Sheets client
    """
    try:
        print_status("Initializing Google Sheets client from dict...", "process")
        
        # Define the scope for Google Sheets API
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        print_status("API scopes configured", "success")
        
        # Load credentials from dictionary
        # Make sure to set GOOGLE_SERVICE_ACCOUNT_INFO as a dict in your settings
        print_status("Loading credentials from dictionary...", "process")
        credentials = Credentials.from_service_account_info(
            settings.GOOGLE_SERVICE_ACCOUNT_INFO, 
            scopes=scope
        )
        print_status("Credentials loaded from dictionary", "success")
        
        # Create and return the client
        print_status("Authorizing Google Sheets client...", "process")
        client = gspread.authorize(credentials)
        print_status("Google Sheets client authorized successfully", "success")
        return client
    
    except Exception as e:
        print_status(f"Failed to create Google Sheets client from dict: {e}", "error")
        logger.error(f"Error creating Google Sheets client from dict: {e}")
        return None

def load_master_from_google_sheets():
    """
    Load master data from Google Sheets and return as a dictionary of DataFrames.
    
    Returns:
        dict: Dictionary with sheet names as keys and DataFrames as values
        None: If loading fails
    """
    try:
        print_status("Starting master data load from Google Sheets...", "process")
        
        # Get the Google Sheets client
        client = get_google_sheets_client()
        if not client:
            print_status("Google Sheets client creation failed", "error")
            logger.error("Failed to create Google Sheets client")
            return None
        
        # Open the spreadsheet using the ID from settings
        print_status("Opening spreadsheet...", "process")
        spreadsheet = client.open_by_key(settings.MASTER_DATA_SHEET_ID)
        print_status(f"Spreadsheet opened: {spreadsheet.title}", "success")
        
        # Get all worksheets
        print_status("Retrieving worksheets...", "process")
        worksheets = spreadsheet.worksheets()
        print_status(f"Found {len(worksheets)} worksheets", "success")
        
        master_data = {}
        
        for worksheet in worksheets:
            try:
                print_status(f"Processing worksheet: {worksheet.title}", "process")
                
                # Get all values from the worksheet
                values = worksheet.get_all_values()
                
                if not values:
                    print_status(f"Worksheet '{worksheet.title}' is empty", "warning")
                    logger.warning(f"Worksheet '{worksheet.title}' is empty")
                    continue
                
                # Convert to DataFrame
                # First row is assumed to be headers
                headers = values[0]
                data = values[1:] if len(values) > 1 else []
                
                df = pd.DataFrame(data, columns=headers)
                
                # Clean up empty rows and columns
                df = df.dropna(how='all')  # Remove completely empty rows
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
                
                # Convert numeric columns where possible
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                
                master_data[worksheet.title] = df
                print_status(f"Worksheet '{worksheet.title}' loaded successfully ({len(df)} rows)", "success")
                logger.info(f"Successfully loaded worksheet '{worksheet.title}' with {len(df)} rows")
                
            except Exception as e:
                print_status(f"Failed to load worksheet '{worksheet.title}': {e}", "error")
                logger.error(f"Error loading worksheet '{worksheet.title}': {e}")
                continue
        
        if master_data:
            print_status(f"Master data load completed - {len(master_data)} worksheets loaded", "success")
            logger.info(f"Successfully loaded {len(master_data)} worksheets from Google Sheets")
            return master_data
        else:
            print_status("No worksheets were successfully loaded", "error")
            logger.error("No worksheets were successfully loaded")
            return None
            
    except Exception as e:
        print_status(f"Master data load failed: {e}", "error")
        logger.error(f"Error loading master data from Google Sheets: {e}")
        return None

def refresh_master_data_cache(request):
    """
    Refresh the master data cache by loading fresh data from Google Sheets.
    
    Args:
        request: Django request object (for session access)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print_status("Starting master data cache refresh...", "process")
        
        master25 = load_master_from_google_sheets()
        if master25:
            print_status("Converting data to JSON for session storage...", "process")
            request.session['master25'] = {
                name: df.to_json(date_format='iso') for name, df in master25.items()
            }
            print_status("Master data cache refreshed successfully", "success")
            logger.info("Master data cache refreshed successfully")
            return True
        else:
            print_status("Master data cache refresh failed", "error")
            logger.error("Failed to refresh master data cache")
            return False
    except Exception as e:
        print_status(f"Cache refresh error: {e}", "error")
        logger.error(f"Error refreshing master data cache: {e}")
        return False

def get_sheet_info(sheet_id):
    """
    Get basic information about a Google Sheet.
    
    Args:
        sheet_id (str): Google Sheets ID
    
    Returns:
        dict: Sheet information including title, worksheets, etc.
        None: If operation fails
    """
    try:
        print_status("Retrieving sheet information...", "process")
        
        client = get_google_sheets_client()
        if not client:
            print_status("Failed to get Google Sheets client", "error")
            return None
        
        print_status("Opening spreadsheet for info retrieval...", "process")
        spreadsheet = client.open_by_key(sheet_id)
        print_status(f"Spreadsheet accessed: {spreadsheet.title}", "success")
        
        info = {
            'title': spreadsheet.title,
            'id': spreadsheet.id,
            'url': spreadsheet.url,
            'worksheets': []
        }
        
        print_status("Collecting worksheet information...", "process")
        for worksheet in spreadsheet.worksheets():
            worksheet_info = {
                'title': worksheet.title,
                'id': worksheet.id,
                'row_count': worksheet.row_count,
                'col_count': worksheet.col_count
            }
            info['worksheets'].append(worksheet_info)
            print_status(f"Worksheet info collected: {worksheet.title} ({worksheet.row_count}x{worksheet.col_count})", "success")
        
        print_status("Sheet information retrieval completed", "success")
        return info
    
    except Exception as e:
        print_status(f"Failed to get sheet info: {e}", "error")
        logger.error(f"Error getting sheet info: {e}")
        return None

def validate_google_sheets_connection():
    """
    Validate that Google Sheets API connection is working.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        print_status("Validating Google Sheets connection...", "process")
        
        client = get_google_sheets_client()
        if not client:
            print_status("Google Sheets client validation failed", "error")
            return False, "Failed to create Google Sheets client"
        
        # Try to access the master sheet
        print_status("Testing access to master sheet...", "process")
        spreadsheet = client.open_by_key(settings.MASTER_DATA_SHEET_ID)
        
        success_msg = f"Successfully connected to sheet: {spreadsheet.title}"
        print_status(success_msg, "success")
        return True, success_msg
    
    except Exception as e:
        error_msg = f"Connection failed: {str(e)}"
        print_status(error_msg, "error")
        return False, error_msg

def error_fix(df):
    df = df.copy() 
    to_drop = set()
    col = df.columns[1]

    i = 0
    while i <= len(df) - 3:
        idx_i = df.index[i]
        idx_ip1 = df.index[i+1]
        idx_ip2 = df.index[i+2]

        if idx_i in to_drop or idx_ip1 in to_drop or idx_ip2 in to_drop:
            i += 1
            continue

        v_i = df.loc[idx_i, col]
        v_ip1 = df.loc[idx_ip1, col]
        v_ip2 = df.loc[idx_ip2, col]

        if v_i > v_ip1 and v_i > v_ip2:
            to_drop.add(idx_i)
        elif v_i > v_ip1 and v_i < v_ip2:
            to_drop.add(idx_ip1)
        i += 1

    # Additional rule: second last > last
    if len(df) >= 2:
        second_last_idx = df.index[-2]
        last_idx = df.index[-1]
        if df.loc[second_last_idx, col] > df.loc[last_idx, col]:
            to_drop.add(second_last_idx)

    return df.drop(index=to_drop).reset_index(drop=True)

def get_filled(meter_name, filtered_data, date_column, start_date, end_date, plot_size):
    if not meter_name:
        raise ValueError("Meter name missing.")
        
    # Filter data for the specific meter
    df = filtered_data[filtered_data['Meter Serial Number - as shown on meter'] == meter_name].copy()
    if df.empty:
        return pd.DataFrame()
        
    # Keep only relevant columns
    df = df[[date_column, 'Reading in the meter - in m3']].dropna()

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort by date (if timestamps existed, include them in sort)
    df = df.sort_values(date_column)

    # Deduplicate: keep the latest reading if multiple on same date
    df = df.drop_duplicates(subset=[date_column], keep='last')

    df = error_fix(df)

    # Calculate time difference and deltas
    df['Days Since Previous Reading'] = df[date_column].diff().dt.days.fillna(method='bfill').fillna(1).astype(int)
    df['Delta m³'] = df['Reading in the meter - in m3'].diff().fillna(method='bfill').fillna(0)

    if plot_size <= 0:
        raise ValueError("Invalid plot size.")

    # Normalize per acre and per day
    df['m³ per Acre'] = df['Delta m³'] / plot_size
    df['m³ per Acre per Avg Day'] = df['m³ per Acre'] / df['Days Since Previous Reading'].replace(0, 1)


    # Create filled daily dataframe
    filled = create_adjusted_filled_dataframe(
        meter_data=df,
        start_date=start_date,
        end_date=end_date,
        date_column=date_column,
        avg_day_column='m³ per Acre per Avg Day'
    )

    return filled

def create_adjusted_filled_dataframe(meter_data, start_date, end_date, date_column, avg_day_column):
    # shift first reading to align with start_date
    filtered = meter_data[meter_data[date_column] >= start_date]

    if filtered.empty:
        return filtered

    if not filtered.empty:
        first_index = filtered.index[0]
        days_diff = (filtered.iloc[0][date_column] - start_date).days + 1
            
        # Get column names explicitly
        col_m3_per_acre = 'm³ per Acre'
        col_avg_day = 'm³ per Acre per Avg Day'

        # Update values explicitly using column names
        meter_data.loc[first_index, col_m3_per_acre] = meter_data.loc[first_index, col_m3_per_acre]
        meter_data.loc[first_index, col_avg_day] = meter_data.loc[first_index, col_avg_day] / days_diff

        
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    base = pd.DataFrame({date_column: all_dates})
    merged = pd.merge(base, meter_data, on=date_column, how='left')

    last_date = start_date
    for idx, row in merged.iterrows():
        if pd.isna(row[avg_day_column]):
            next_rows = meter_data[meter_data[date_column] > row[date_column]]
            if not next_rows.empty:
                merged.iloc[idx, 1:] = next_rows.iloc[0, 1:]
            else:
                merged.iloc[idx, 1:] = 0
        else:
            last_date = row[date_column]
    return merged

def kharif2024_farms(master_df):
    meters_df = master_df['Meters'][['Kharif 24 Meter Serial No', 'Kharif 24 FARMID']].copy()
    meters_df = meters_df.dropna(subset=['Kharif 24 Meter Serial No', 'Kharif 24 FARMID'])
    meters_df = meters_df[
        (meters_df['Kharif 24 Meter Serial No'].astype(str).str.strip() != '') &
        (meters_df['Kharif 24 FARMID'].astype(str).str.strip() != '')
    ]
    farm_to_meters = meters_df.groupby('Kharif 24 FARMID')['Kharif 24 Meter Serial No'].apply(list)
    return farm_to_meters.to_dict()


def encode_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def get_2024plots(selected_farm, meter_df, master_df, meter_list):
    kharif_df = master_df.get('Kharif 24')
    if kharif_df is None:
        raise ValueError("Kharif 24 sheet not found in master data.")

    farm_row = kharif_df[kharif_df['Kharif 24 FarmID'] == selected_farm]
    if farm_row.empty:
        raise ValueError(f"Farm ID {selected_farm} missing in Kharif 24 data.")

    # determine start/end dates
    if pd.notna(farm_row['Kharif 24 Paddy transplanting date (TPR)'].values[0]):
        start_date = pd.to_datetime(
            farm_row['Kharif 24 Paddy transplanting date (TPR)'].values[0],
            dayfirst=True
        )
    elif pd.notna(farm_row['Kharif 24 Paddy sowing (DSR)'].values[0]):
        start_date = pd.to_datetime(
            farm_row['Kharif 24 Paddy sowing (DSR)'].values[0],
            dayfirst=True
        )
    else:
        start_date = pd.to_datetime('15/06/2024', dayfirst=True)

    if pd.notna(farm_row['Kharif 24 Paddy Harvest date'].values[0]):
        end_date = pd.to_datetime(
            farm_row['Kharif 24 Paddy Harvest date'].values[0],
            dayfirst=True
        )
    else:
        end_date = pd.to_datetime('31/10/2024', dayfirst=True)

    # clean and filter meter data
    raw = meter_df.dropna(axis=1, how='all')
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce').dt.date
    reading_cols = [c for c in raw.columns if 'reading' in c.lower()]
    filtered = raw[[date_col] + reading_cols].dropna(subset=reading_cols, how='all')

    # get filled dataframes per meter
    plot_size = farm_row['Kharif 24 Acres farm/plot'].values[0]
    filled = {
        m: get_filled_2024(m, filtered, date_col, start_date, end_date, plot_size)
        for m in meter_list
    }

    plots = []
    for m, df in filled.items():
        df['Day'] = (pd.to_datetime(df[date_col]) - start_date).dt.days

        # m³ per Acre
        fig1, ax1 = plt.subplots()
        ax1.plot(df['Day'], df['m³ per Acre'])
        ax1.set(xlabel='Days from start', ylabel='m³ per Acre',
                title=f'm³ per Acre | Meter {m} | Farm {selected_farm}')
        fig1.tight_layout()
        plots.append(fig1)

        # m³ per Acre per Avg Day
        fig2, ax2 = plt.subplots()
        ax2.plot(df['Day'], df['m³ per Acre per Avg Day'])
        ax2.set(xlabel='Days from start', ylabel='m³ per Acre per Avg Day',
                title=f'm³ per Acre per Avg Day | Meter {m} | Farm {selected_farm}')
        fig2.tight_layout()
        plots.append(fig2)

    return plots


def create_adjusted_filled_dataframe_2024(meter_data, start_date, end_date, date_column, avg_day_column):
    # shift first reading to align with start_date
    filtered = meter_data[meter_data[date_column] > start_date]
    if not filtered.empty:
        days_diff = (filtered.iloc[0, 0] - start_date).days + 1

        meter_data.loc[
            filtered.index[0],
            meter_data.columns[-2:]
        ] = [
            meter_data.iloc[filtered.index[0], -2],
            meter_data.iloc[filtered.index[0], -1] / days_diff
        ]

    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    base = pd.DataFrame({date_column: all_dates})
    merged = pd.merge(base, meter_data, on=date_column, how='left')

    last_date = start_date
    for idx, row in merged.iterrows():
        if pd.isna(row[avg_day_column]):
            next_rows = meter_data[meter_data[date_column] > row[date_column]]
            if not next_rows.empty:
                merged.iloc[idx, 2:] = next_rows.iloc[0, 2:]
            else:
                merged.iloc[idx, 2:] = 0
        else:
            last_date = row[date_column]

    return merged


def get_filled_2024(meter_name, filtered_data, date_column, start_date, end_date, plot_size):
    if not meter_name:
        raise ValueError("Meter name missing.")

    target = f"{meter_name} - reading"
    cols = [c for c in filtered_data.columns if c == date_column or target in c]
    df = filtered_data[cols].dropna(subset=[cols[-1]]).sort_values(date_column)
    df[date_column] = pd.to_datetime(df[date_column])
    df['Days Since Previous Reading'] = df[date_column].diff().dt.days.fillna(method='bfill').astype(int)
    df['Delta m³'] = df[cols[-1]].diff().fillna(method='bfill')

    if plot_size <= 0:
        raise ValueError("Invalid plot size.")

    df['m³ per Acre'] = df['Delta m³'] / plot_size
    df['m³ per Acre per Avg Day'] = df['m³ per Acre'] / df['Days Since Previous Reading'].replace(0, 1)

    return create_adjusted_filled_dataframe_2024(
        meter_data=df,
        start_date=start_date,
        end_date=end_date,
        date_column=date_column,
        avg_day_column='m³ per Acre per Avg Day'
    )


# --- 2025 Helpers ---

def kharif2025_farms(master_df):
    sheet = master_df.get('Farm details')
    if sheet is None:
        raise ValueError("‘Kharif 25’ sheet not found.")
    mapping = {}
    for _, row in sheet.iterrows():
        farm = row.get('Kharif 25 Farm ID')

        if pd.isna(farm):
            continue
        meters = []
        for col in ['Kharif 25 Meter serial number - 1', 'Kharif 25 Meter serial number - 2']:
            serial = row.get(col)
            if pd.notna(serial):
                meters.append(str(serial).strip())
        if meters:
            mapping[str(farm)] = meters

    return mapping


def get_2025plots(raw_df, master_df, selected_farm, meter_list, start_date_enter = None, end_date_enter = None):
    meta = master_df['Farm details']
    farm_row = meta[meta['Kharif 25 Farm ID'] == selected_farm]
    if farm_row.empty:
        raise ValueError(f"Farm ID {selected_farm} not found in metadata.")
    farm_row = farm_row.iloc[0]

    acreage = farm_row.get('Kharif 25 Acres farm - farmer reporting') or 1
    if pd.isna(acreage) or acreage <= 0:
        acreage = 1
        print("acreage is nan")
    if pd.notna(farm_row.get('Kharif 25 Paddy transplanting date (TPR)')):
        start_date = pd.to_datetime(farm_row['Kharif 25 Paddy transplanting date (TPR)'], dayfirst=True)
    else:
        start_date = pd.to_datetime('20/06/2025', dayfirst=True)

    if start_date_enter is not None:
        start_date = start_date_enter

    # Clean meter data
    date_col = 'Date'
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
    raw_df = raw_df.sort_values(date_col)

    plots_base64 = []

    for meter in meter_list:
        end_date = pd.to_datetime(datetime.now().date())
        if end_date_enter is not None:
            end_date = end_date_enter
        
        filled_df = get_filled(meter, raw_df, date_col, start_date, end_date, acreage)
        if filled_df.empty:
            continue

        filled_df['Day'] = (pd.to_datetime(filled_df[date_col]) - start_date).dt.days
        filled_df['7-day SMA'] = filled_df['m³ per Acre per Avg Day'].rolling(window=7, min_periods=1).mean()

        # List to collect plots for this meter
        meter_plots = []

        # --- Graph 1: m3 per acre per avg day ---
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(filled_df['Day'], filled_df['m³ per Acre per Avg Day'], label='Daily m³ per Acre per day', color='blue')
        ax1.set(title=f'Daily Avg per Acre | Meter {meter}', xlabel='Days from transplanting', ylabel='Daily Avg m³ per Acre')
        ax1.legend()
        # Force axis to start from zero
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        fig1.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig1))
        plt.close(fig1)

        # --- Graph 2: Moving Averages Comparison ---
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(filled_df['Day'], filled_df['7-day SMA'], label='7-day SMA', linestyle='--', color='green')
        # ax2.plot(filled_df['Day'], filled_df['Weekly Avg'], label='Weekly Avg', linestyle=':', color='orange')
        ax2.set(title=f'Moving Average | Meter {meter}', xlabel='Days from transplanting', ylabel='7-days SMA m³ per Acre')
        ax2.legend()
        # Force axis to start from zero
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        fig2.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig2))
        plt.close(fig2)

        # --- Graph 3: Delta Analysis ---
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(filled_df['Day'], filled_df['Delta m³'], marker='o', linestyle='-', color='purple', label='Delta m³')
        ax3.set(title=f'Meter Actual readings of meter | Meter {meter}', xlabel='Days from transplanting', ylabel='Delta of readings (m³)')
        ax3.legend()
        # Force axis to start from zero
        ax3.set_xlim(left=0)
        ax3.set_ylim(bottom=0)
        fig3.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig3))
        plt.close(fig3)

        # --- Graph 4: Delta per Acre ---
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(filled_df['Day'], filled_df['m³ per Acre'], marker='x', linestyle='-', color='red', label='Delta m³/Acre')
        ax4.set(title=f'Meter readings per Acre | Meter {meter}', xlabel='Days from transplanting', ylabel='Delta of reading per Acre (m³) ')
        ax4.legend()
        # Force axis to start from zero
        ax4.set_xlim(left=0)
        ax4.set_ylim(bottom=0)
        fig4.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig4))
        plt.close(fig4)

        plots_base64.extend(meter_plots)

    return plots_base64

def get_2025plots_combined(raw_df, master_df, selected_farm, meter_list, start_date_enter=None, end_date_enter=None):
    """
    Generate combined plots for multiple meters treating them as a single unit
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import base64
    import io
    
    if len(meter_list) < 2:
        return []  # No combined plots for single meter
    
    def encode_plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Get farm metadata
    meta = master_df['Farm details']
    farm_row = meta[meta['Kharif 25 Farm ID'] == selected_farm]
    if farm_row.empty:
        raise ValueError(f"Farm ID {selected_farm} not found in metadata.")
    farm_row = farm_row.iloc[0]
    
    acreage = farm_row.get('Kharif 25 Acres farm - farmer reporting') or 1
    if pd.isna(acreage) or acreage <= 0:
        acreage = 1
    
    if pd.notna(farm_row.get('Kharif 25 Paddy transplanting date (TPR)')):
        start_date = pd.to_datetime(farm_row['Kharif 25 Paddy transplanting date (TPR)'], dayfirst=True)
    else:
        start_date = pd.to_datetime('20/06/2025', dayfirst=True)
    
    if start_date_enter is not None:
        start_date = pd.to_datetime(start_date_enter)
    
    end_date = pd.to_datetime(datetime.now().date())
    if end_date_enter is not None:
        end_date = pd.to_datetime(end_date_enter)
    
    # Clean meter data
    date_col = 'Date'
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
    raw_df = raw_df.sort_values(date_col)
    
    # Get data for all meters
    all_meter_data = []
    for meter in meter_list:
        meter_df = raw_df[raw_df['Meter Serial Number - as shown on meter'] == meter].copy()
        if not meter_df.empty:
            meter_df = meter_df[[date_col, 'Reading in the meter - in m3']].dropna()
            meter_df[date_col] = pd.to_datetime(meter_df[date_col])
            meter_df = meter_df.sort_values(date_col)
            meter_df = meter_df.drop_duplicates(subset=[date_col], keep='last')
            meter_df['Meter'] = meter
            all_meter_data.append(meter_df)
    
    if not all_meter_data:
        return []
    
    # Combine all meter data
    combined_df = pd.concat(all_meter_data, ignore_index=True)
    
    # Group by date and sum readings
    combined_grouped = combined_df.groupby(date_col).agg({
        'Reading in the meter - in m3': 'sum'
    }).reset_index()
    
    # Calculate deltas for combined data
    combined_grouped = combined_grouped.sort_values(date_col)
    combined_grouped['Days Since Previous Reading'] = combined_grouped[date_col].diff().dt.days.fillna(1).astype(int)
    combined_grouped['Delta m³'] = combined_grouped['Reading in the meter - in m3'].diff().fillna(0)
    
    # Normalize per acre
    combined_grouped['m³ per Acre'] = combined_grouped['Delta m³'] / acreage
    combined_grouped['m³ per Acre per Avg Day'] = combined_grouped['m³ per Acre'] / combined_grouped['Days Since Previous Reading'].replace(0, 1)
    
    # Fill daily data
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    base = pd.DataFrame({date_col: all_dates})
    filled_df = pd.merge(base, combined_grouped, on=date_col, how='left')
    
    # Forward fill values
    for col in ['m³ per Acre', 'm³ per Acre per Avg Day', 'Delta m³']:
        filled_df[col] = filled_df[col].fillna(0)
    
    # Calculate days from start
    filled_df['Day'] = (filled_df[date_col] - start_date).dt.days
    filled_df['7-day SMA'] = filled_df['m³ per Acre per Avg Day'].rolling(window=7, min_periods=1).mean()
    
    # Generate plots
    plots_base64 = []
    meters_label = " + ".join(meter_list)
    
    # Graph 1: Daily avg per acre
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(filled_df['Day'], filled_df['m³ per Acre per Avg Day'], label='Combined Daily m³ per Acre per day', color='blue', linewidth=2)
    ax1.set(title=f'Combined Daily Avg per Acre | Meters: {meters_label}', 
            xlabel='Days from transplanting', 
            ylabel='Daily Avg m³ per Acre')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plots_base64.append(encode_plot_to_base64(fig1))
    plt.close(fig1)
    
    # Graph 2: 7-day SMA
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(filled_df['Day'], filled_df['7-day SMA'], label='Combined 7-day SMA', linestyle='--', color='green', linewidth=2)
    ax2.set(title=f'Combined Moving Average | Meters: {meters_label}', 
            xlabel='Days from transplanting', 
            ylabel='7-days SMA m³ per Acre')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plots_base64.append(encode_plot_to_base64(fig2))
    plt.close(fig2)
    
    # Graph 3: Delta Analysis
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(filled_df['Day'], filled_df['Delta m³'], marker='o', linestyle='-', color='purple', label='Combined Delta m³', linewidth=2)
    ax3.set(title=f'Combined Meter Readings | Meters: {meters_label}', 
            xlabel='Days from transplanting', 
            ylabel='Combined Delta of readings (m³)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    plots_base64.append(encode_plot_to_base64(fig3))
    plt.close(fig3)
    
    # Graph 4: Delta per Acre
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(filled_df['Day'], filled_df['m³ per Acre'], marker='x', linestyle='-', color='red', label='Combined Delta m³/Acre', linewidth=2)
    ax4.set(title=f'Combined Readings per Acre | Meters: {meters_label}', 
            xlabel='Days from transplanting', 
            ylabel='Combined Delta per Acre (m³)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    plots_base64.append(encode_plot_to_base64(fig4))
    plt.close(fig4)
    
    return plots_base64

def get_tables(raw_df, master_df, farm_list, col_to_get, start_date_enter = None, end_date_enter = None):
    # Clean meter data
    date_col = 'Date'
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
    raw_df = raw_df.sort_values(date_col)
    dfs = []
    
    for farm in farm_list.keys():
        # Clean meter data
        date_col = 'Date'
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
        raw_df = raw_df.sort_values(date_col)

        meta = master_df['Farm details']
        farm_row = meta[meta['Kharif 25 Farm ID'] == farm]
        if farm_row.empty:
            raise ValueError(f"Farm ID {farm} not found in metadata.")
        farm_row = farm_row.iloc[0]

        acreage = farm_row.get('Kharif 25 Acres farm - farmer reporting') or 1
        if pd.isna(acreage) or acreage <= 0:
            acreage = 1
            print("acreage is nan")
        if pd.notna(farm_row.get('Kharif 25 Paddy transplanting date (TPR)')):
            start_date = pd.to_datetime(farm_row['Kharif 25 Paddy transplanting date (TPR)'], dayfirst=True)
        else:
            start_date = pd.to_datetime('20/06/2025', dayfirst=True)
        
        if start_date_enter is not None:
            start_date = start_date_enter

        
        
        for meter in farm_list[farm]:
            end_date = pd.to_datetime(datetime.now().date())
            if end_date_enter is not None:
                end_date = end_date_enter
            filled_df = get_filled(meter, raw_df, date_col, start_date, end_date, acreage)
            if filled_df.empty:
                filled_df = pd.DataFrame(columns=['Day', meter])
            else:
                filled_df['Day'] = (pd.to_datetime(filled_df[date_col]) - start_date).dt.days
                filled_df = filled_df[['Day', col_to_get]].rename(columns={col_to_get: meter})
            
            dfs.append(filled_df)

    from functools import reduce

    # Assuming all DataFrames have the same name for the date column
    combined_df = reduce(lambda left, right: pd.merge(left, right, on='Day', how='outer'), dfs)

    return combined_df


def calculate_avg_m3_per_acre(group_type, group_label, farm_ids, raw_df, master25, column_to_see, start_date_enter = None, end_date_enter = None):
    """
    Given a group label (like 'Group-A Complied') and its farm IDs, 
    returns a dataframe with Days column and average m³ per acre per day.
    """
    kharif_sheet = master25.get("Farm details")
    all_meter_dfs = []


    # --- Start of main function logic ---

    for farm_id in farm_ids:
        # Get acreage
        acreage_row = kharif_sheet[kharif_sheet["Kharif 25 Farm ID"] == farm_id]
        if acreage_row.empty:
            continue
        acreage = acreage_row["Kharif 25 Acres farm - farmer reporting"].values[0]
        if pd.isna(acreage) or acreage == 0:
            acreage = 1
        # Get transplanting date
        tpr_col = "Kharif 25 Paddy transplanting date (TPR)"
        tpr_date = acreage_row[tpr_col].values[0]
        if pd.isna(tpr_date) or tpr_date == "":
            tpr_date = pd.to_datetime("2025-06-20")
        else:
            tpr_date = pd.to_datetime(tpr_date)
        
        if start_date_enter is not None:
            tpr_date = start_date_enter

        # Get meter serial numbers
        meters = []
        for m_col in ["Kharif 25 Meter serial number - 1", "Kharif 25 Meter serial number - 2"]:
            val = acreage_row[m_col].values[0]
            if pd.notna(val) and val != "":
                meters.append(str(val).strip())

        # Clean meter data
        date_col = 'Date'
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
        raw_df = raw_df.sort_values(date_col)


        # Process each meter
        for meter in meters:
            end_date = pd.to_datetime(datetime.now().date())
          
            if end_date_enter is not None:
                end_date = end_date_enter
            filled_df = get_filled(meter, raw_df, date_col, tpr_date, end_date, acreage)
            if filled_df.empty:
                continue
            filled_df['Day'] = (pd.to_datetime(filled_df[date_col]) - tpr_date).dt.days
            meter_df = filled_df[["Day", column_to_see]].reset_index(drop=True).rename(columns={column_to_see: meter})
            all_meter_dfs.append(meter_df)

    if not all_meter_dfs:
        return pd.DataFrame(columns=["Day", group_label])
    

    # Merge on Days
    merged = pd.DataFrame()
    from functools import reduce

    # Assuming all DataFrames have the same name for the date column
    merged = reduce(lambda left, right: pd.merge(left, right, on='Day', how='outer'), all_meter_dfs)

    # Average across all meters day-wise
    avg_series = merged.drop(columns=["Day"]).mean(axis=1)
    final_df = pd.DataFrame({
        "Day": merged["Day"],
        group_label: avg_series
    })

    return final_df

def generate_group_analysis_plot(df, col_name):
    """
    Takes a DataFrame where:
    - The first column is 'Day'
    - All other columns are group names, with average m³/acre/day values
    Returns base64-encoded image string
    """
    plt.figure(figsize=(10, 6))
    for col in df.columns[1:]:
        plt.plot(df['Day'], df[col], label=col, linewidth=2)
    
    plt.xlabel("Days from transplanting")
    plt.ylabel(col_name)
    plt.title("Group-wise Water Usage Comparison")
    plt.grid(True)
    plt.legend()
    # Force axis to start from zero using current axes
    ax = plt.gca()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded


        

        
def get_meters_by_village(raw_df, village_name):
    """
    Get all unique meter serial numbers for a specific village.
    Village name is in column D (index 3) of the raw data.
    """
    # Filter rows where village matches
    village_meters = raw_df[raw_df.iloc[:, 3] == village_name]['Meter Serial Number - as shown on meter'].unique()
    return [str(m).strip() for m in village_meters if pd.notna(m)]


def generate_word_report(results, filter_type, filter_value, raw_df, master_df):
    """
    Generate a Word document report with graphs and details
    """
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    import base64
    from datetime import datetime
    
    # Create document
    doc = Document()
    
    # Set up styles
    # Title style
    title_style = doc.styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
    title_style.font.name = 'Arial'
    title_style.font.size = Pt(24)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 139)  # Dark blue
    
    # Heading style
    heading_style = doc.styles.add_style('CustomHeading', WD_STYLE_TYPE.PARAGRAPH)
    heading_style.font.name = 'Arial'
    heading_style.font.size = Pt(16)
    heading_style.font.bold = True
    heading_style.font.color.rgb = RGBColor(0, 0, 139)
    
    # Add title
    title = doc.add_paragraph('DIGI-VI Water Meter Analysis Report', style='CustomTitle')
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add date
    date_para = doc.add_paragraph(f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()  # Empty line
    
    # Add filter information
    filter_heading = doc.add_paragraph('Analysis Filter Information', style='CustomHeading')
    
    filter_info = doc.add_paragraph()
    filter_info.add_run('Filter Type: ').bold = True
    filter_info.add_run(f'{filter_type}\n')
    filter_info.add_run('Filter Value: ').bold = True
    filter_info.add_run(f'{filter_value}\n')
    
    if filter_type == "Village Filter":
        # Count total meters in village
        village_meters = raw_df[raw_df.iloc[:, 3] == filter_value]['Meter Serial Number - as shown on meter'].nunique()
        filter_info.add_run('Total Meters in Village: ').bold = True
        filter_info.add_run(f'{village_meters}\n')
    
    doc.add_page_break()
    
    # Process each meter's results
    for idx, block in enumerate(results):
        # Add meter heading
        meter_heading = doc.add_paragraph(f'Meter Unit: {block["meter"]}', style='CustomHeading')
        
        # Add meter details
        details = doc.add_paragraph()
        if 'farm' in block and block['farm']:
            details.add_run('Associated Farm ID: ').bold = True
            details.add_run(f'{block["farm"]}\n')
            
            # Get additional farm details from master data
            farm_details = master_df['Farm details']
            farm_row = farm_details[farm_details['Kharif 25 Farm ID'] == block['farm']]
            if not farm_row.empty:
                farm_row = farm_row.iloc[0]
                
                # Add village name
                village_col = 'Kharif 25 Village'
                if village_col in farm_row and pd.notna(farm_row[village_col]):
                    details.add_run('Village: ').bold = True
                    details.add_run(f'{farm_row[village_col]}\n')
                
                # Add acreage
                acre_col = 'Kharif 25 Acres farm - farmer reporting'
                if acre_col in farm_row and pd.notna(farm_row[acre_col]):
                    details.add_run('Farm Size: ').bold = True
                    details.add_run(f'{farm_row[acre_col]} acres\n')
                
                # Add transplanting date
                tpr_col = 'Kharif 25 Paddy transplanting date (TPR)'
                if tpr_col in farm_row and pd.notna(farm_row[tpr_col]):
                    details.add_run('Transplanting Date: ').bold = True
                    details.add_run(f'{farm_row[tpr_col]}\n')
        
        doc.add_paragraph()  # Empty line
        
        # Add graphs
        graph_titles = [
            "Daily Average Water Usage per Acre",
            "7-Day Moving Average Pattern",
            "Meter Actual Readings (Delta)",
            "Water Usage per Acre (Delta)"
        ]
        
        graph_descriptions = [
            "This graph shows the daily average water consumption in cubic meters per acre per day. It helps identify daily irrigation patterns and water usage efficiency.",
            "This graph displays the 7-day simple moving average, smoothing out daily fluctuations to reveal longer-term trends in water usage.",
            "This graph presents the actual meter reading differences (delta) between consecutive readings, showing the total water consumed between measurements.",
            "This graph normalizes the water consumption per acre, making it easier to compare water usage across farms of different sizes."
        ]
        
        for graph_idx, img_base64 in enumerate(block['plots']):
            # Add graph title
            graph_title = doc.add_paragraph()
            graph_title.add_run(f'Graph {graph_idx + 1}: {graph_titles[graph_idx]}').bold = True
            
            # Add graph description
            doc.add_paragraph(graph_descriptions[graph_idx])
            
            # Decode and add image
            try:
                img_data = base64.b64decode(img_base64)
                img_stream = BytesIO(img_data)
                doc.add_picture(img_stream, width=Inches(6))
                doc.add_paragraph()  # Empty line after image
            except Exception as e:
                doc.add_paragraph(f"[Error loading graph: {str(e)}]")
        
        # Add page break after each meter (except last)
        if idx < len(results) - 1:
            doc.add_page_break()
    
    # Add footer
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.add_run('--- End of Report ---').italic = True
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Save to BytesIO
    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    
    return docx_buffer



def generate_group_analysis_report(group_type, selected_groups, group_plot_base64, group_plot2, group_data):
    """
    Generate a Word document report for group analysis
    """
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    import base64
    from datetime import datetime
    
    # Create document
    doc = Document()
    
    # Set up styles
    title_style = doc.styles.add_style('CustomTitle', WD_STYLE_TYPE.PARAGRAPH)
    title_style.font.name = 'Arial'
    title_style.font.size = Pt(24)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 139)
    
    heading_style = doc.styles.add_style('CustomHeading', WD_STYLE_TYPE.PARAGRAPH)
    heading_style.font.name = 'Arial'
    heading_style.font.size = Pt(16)
    heading_style.font.bold = True
    heading_style.font.color.rgb = RGBColor(0, 0, 139)
    
    # Add title
    title = doc.add_paragraph('DIGI-VI Group Analysis Report', style='CustomTitle')
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add date
    date_para = doc.add_paragraph(f'Generated on: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}')
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    
    # Add analysis information
    analysis_heading = doc.add_paragraph('Group Analysis Configuration', style='CustomHeading')
    
    # Group type information
    group_info = doc.add_paragraph()
    group_info.add_run('Study Type: ').bold = True
    group_info.add_run(f'{group_type}\n')
    
    # Selected groups
    group_info.add_run('Selected Groups: ').bold = True
    group_info.add_run(f'{", ".join(selected_groups)}\n')
    
    # Add group descriptions based on type
    doc.add_paragraph()
    desc_heading = doc.add_paragraph('Group Descriptions', style='CustomHeading')
    
    if group_type == "Remote":
        descriptions = {
            "Group-A Complied": "Treatment group with remote controllers who complied with irrigation scheduling",
            "Group-A Non-Complied": "Treatment group with remote controllers who did not comply with scheduling",
            "Group-B Complied": "Control group who complied with traditional irrigation practices",
            "Group-B Non-Complied": "Control group who did not comply with traditional practices"
        }
    elif group_type == "AWD":
        descriptions = {
            "Group-A Complied": "Treatment group practicing Alternate Wetting and Drying (AWD) with compliance",
            "Group-A Non-Complied": "Treatment group assigned AWD but with non-compliance",
            "Group-B Complied": "Control group B with compliance to assigned practices",
            "Group-B Non-Complied": "Control group B with non-compliance",
            "Group-C Complied": "Control group C with compliance to assigned practices",
            "Group-C Non-Complied": "Control group C with non-compliance"
        }
    else:  # TPR/DSR
        descriptions = {
            "TPR": "Transplanted Rice (TPR) - Traditional method of rice cultivation",
            "DSR": "Direct Seeded Rice (DSR) - Water-efficient method of rice cultivation"
        }
    
    for group in selected_groups:
        full_group_name = f"{group_type} {group}" if group_type != "TPR/DSR" else group
        if group in descriptions:
            para = doc.add_paragraph()
            para.add_run(f'{full_group_name}: ').bold = True
            para.add_run(descriptions[group])
    
    doc.add_page_break()
    
    # Add analysis results
    results_heading = doc.add_paragraph('Comparative Analysis Results', style='CustomHeading')
    
    # Add explanation
    doc.add_paragraph(
        "The following graph shows the daily average water usage (m³/acre) comparison between "
        "the selected groups over the cultivation period. This visualization helps identify "
        "water usage patterns and efficiency differences between different farming practices."
    )
    
    doc.add_paragraph()
    
    # Add the comparative graph
    try:
        img_data = base64.b64decode(group_plot_base64)
        img_stream = BytesIO(img_data)
        doc.add_picture(img_stream, width=Inches(6.5))

        img_data = base64.b64decode(group_plot2)
        img_stream = BytesIO(img_data)
        doc.add_picture(img_stream, width=Inches(6.5))
    except Exception as e:
        doc.add_paragraph(f"[Error loading graph: {str(e)}]")
    
    # Add data summary if available
    if group_data:
        doc.add_page_break()
        summary_heading = doc.add_paragraph('Statistical Summary', style='CustomHeading')
        
        # Add summary statistics for each group
        for group_label, farms in group_data.items():
            if farms:  # Only if there are farms in this group
                para = doc.add_paragraph()
                para.add_run(f'{group_label}:').bold = True
                para.add_run(f'\n  • Number of farms: {len(farms)}')
                para.add_run(f'\n  • Farm IDs: {", ".join(farms[:5])}{"..." if len(farms) > 5 else ""}')
    
    # Add footer
    doc.add_paragraph()
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.add_run('--- End of Group Analysis Report ---').italic = True
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Save to BytesIO
    docx_buffer = BytesIO()
    doc.save(docx_buffer)
    docx_buffer.seek(0)
    
    return docx_buffer