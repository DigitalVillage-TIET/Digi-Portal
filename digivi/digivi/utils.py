# digivi/utils.py

import pandas as pd
import io
import base64
import matplotlib.pyplot as plt


# --- 2024 Helpers ---

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


def get_2025plots(raw_df, master_df, selected_farm, meter_list):
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
        start_date = pd.to_datetime('15/05/2025', dayfirst=True)

    # Clean meter data
    date_col = 'Date'
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], dayfirst=False)
    raw_df = raw_df.sort_values(date_col)

    plots_base64 = []

    for meter in meter_list:
        end_date = pd.to_datetime('31/05/2025', dayfirst=True)
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


def get_tables(raw_df, master_df, farm_list, col_to_get):

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

        
        end_date = filtered[date_column].iloc[-1]
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
            start_date = pd.to_datetime('15/05/2025', dayfirst=True)

        
        
        for meter in farm_list[farm]:
            end_date = pd.to_datetime('31/05/2025', dayfirst=True)
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


def calculate_avg_m3_per_acre(group_type, group_label, farm_ids, raw_df, master25):
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

        
        end_date = filtered[date_column].iloc[-1]
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
            tpr_date = pd.to_datetime("2025-05-15")
        else:
            tpr_date = pd.to_datetime(tpr_date)

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
            end_date = pd.to_datetime('31/05/2025', dayfirst=True)
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


