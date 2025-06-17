# digivi/utils.py

import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime


# --- 2024 Helpers ---
def filter_data_by_date_range(raw_df, start_date, end_date):
    """
    Filter raw data by date range
    """
    # Ensure Date column is datetime
    date_col = 'Date'
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
    
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Filter the dataframe
    filtered_df = raw_df[(raw_df[date_col] >= start_date) & (raw_df[date_col] <= end_date)].copy()
    
    return filtered_df

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
        m: get_filled(m, filtered, date_col, start_date, end_date, plot_size)
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


def create_adjusted_filled_dataframe(meter_data, start_date, end_date, date_column, avg_day_column):
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


def get_filled(meter_name, filtered_data, date_column, start_date, end_date, plot_size):
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

    return create_adjusted_filled_dataframe(
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
    import pandas as pd
    import matplotlib.pyplot as plt
    import base64
    import io

    def encode_plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def create_adjusted_filled_dataframe(meter_data, start_date, end_date, date_column, avg_day_column):
        # shift first reading to align with start_date
        filtered = meter_data[meter_data[date_column] > start_date]
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


    # --- Start of main function logic ---

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
        fig1.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig1))
        plt.close(fig1)

        # --- Graph 2: Moving Averages Comparison ---
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(filled_df['Day'], filled_df['7-day SMA'], label='7-day SMA', linestyle='--', color='green')
        # ax2.plot(filled_df['Day'], filled_df['Weekly Avg'], label='Weekly Avg', linestyle=':', color='orange')
        ax2.set(title=f'Moving Average | Meter {meter}', xlabel='Days from transplanting', ylabel='7-days SMA m³ per Acre')
        ax2.legend()
        fig2.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig2))
        plt.close(fig2)

        # --- Graph 3: Delta Analysis ---
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(filled_df['Day'], filled_df['Delta m³'], marker='o', linestyle='-', color='purple', label='Delta m³')
        ax3.set(title=f'Meter Actual readings of meter | Meter {meter}', xlabel='Days from transplanting', ylabel='Delta of readings (m³)')
        ax3.legend()
        fig3.tight_layout()
        meter_plots.append(encode_plot_to_base64(fig3))
        plt.close(fig3)

        # --- Graph 4: Delta per Acre ---
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(filled_df['Day'], filled_df['m³ per Acre'], marker='x', linestyle='-', color='red', label='Delta m³/Acre')
        ax4.set(title=f'Meter readings per Acre | Meter {meter}', xlabel='Days from transplanting', ylabel='Delta of reading per Acre (m³) ')
        ax4.legend()
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

    def create_adjusted_filled_dataframe(meter_data, start_date, end_date, date_column, avg_day_column):
        # shift first reading to align with start_date
        filtered = meter_data[meter_data[date_column] > start_date]

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


def calculate_avg_m3_per_acre(group_type, group_label, farm_ids, raw_df, master25, start_date_enter = None, end_date_enter = None):
    """
    Given a group label (like 'Group-A Complied') and its farm IDs, 
    returns a dataframe with Days column and average m³ per acre per day.
    """
    kharif_sheet = master25.get("Farm details")
    all_meter_dfs = []

    import pandas as pd
    import matplotlib.pyplot as plt
    import base64
    import io

    def create_adjusted_filled_dataframe(meter_data, start_date, end_date, date_column, avg_day_column):
        # shift first reading to align with start_date
        filtered = meter_data[meter_data[date_column] > start_date]

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

        # Calculate time difference and deltas
        df['Days Since Previous Reading'] = df[date_column].diff().dt.days.fillna(method='bfill').fillna(0).astype(int)
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
            meter_df = filled_df[["Day", "m³ per Acre per Avg Day"]].reset_index(drop=True).rename(columns={"m³ per Acre per Avg Day": meter})
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


import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_group_analysis_plot(df):
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
    plt.ylabel("Daily Average m3/acre")
    plt.title("Group-wise Water Usage Comparison")
    plt.grid(True)
    plt.legend()
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



def generate_group_analysis_report(group_type, selected_groups, group_plot_base64, group_data):
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