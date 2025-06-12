# digivi/views.py

from django.shortcuts import render
from django.core.files.storage import default_storage
import pandas as pd
from io import BytesIO
from django.http import HttpResponse

from .utils import (
    kharif2024_farms, get_2024plots,
    kharif2025_farms, get_2025plots,
    encode_plot_to_base64, get_tables,
    calculate_avg_m3_per_acre, generate_group_analysis_plot
)


def index(request):
    return render(request, 'index.html')

def landing(request):
    return render(request, 'landing.html')

def meter_reading(request):
    meter_df = master_df = None
    error_message = meter_preview = None
    master_previews = {}
    farm_ids = []
    meter_results = []
    selected_farm = request.POST.get('selected_farm', "")

    if request.method == 'POST':
        # Upload & cache meter file
        if 'meter_file' in request.FILES:
            try:
                meter_df = pd.read_excel(request.FILES['meter_file'])
                date_col = meter_df.columns[0]
                meter_df[date_col] = pd.to_datetime(meter_df[date_col], errors='coerce')
                request.session['meter_df'] = meter_df.to_json(date_format='iso')
                meter_preview = meter_df.head().to_html()
            except Exception as e:
                error_message = f"Error processing meter file: {e}"

        # Upload & cache master file
        if 'master_file' in request.FILES:
            try:
                master = pd.read_excel(request.FILES['master_file'], sheet_name=None)
                request.session['master_df'] = {
                    k: v.to_json(date_format='iso') for k, v in master.items()
                }
                for name, df in master.items():
                    master_previews[name] = df.head().to_html()
            except Exception as e:
                error_message = f"Error processing master file: {e}"

        # Retrieve from session if missing
        if meter_df is None and 'meter_df' in request.session:
            meter_df = pd.read_json(request.session['meter_df'])
        if master_df is None and 'master_df' in request.session:
            master_df = {k: pd.read_json(v) for k, v in request.session['master_df'].items()}
            for name, df in master_df.items():
                master_previews[name] = df.head().to_html()

        # Farm dropdown
        if master_df:
            farms_dict = kharif2024_farms(master_df)
            farm_ids = list(farms_dict.keys())

        # Generate plots
        if selected_farm and master_df:
            figs = get_2024plots(selected_farm, meter_df, master_df, farms_dict[selected_farm])
            plots64 = [encode_plot_to_base64(f) for f in figs]

            kharif = master_df['Kharif 24']
            row = kharif[kharif['Kharif 24 FarmID'] == selected_farm].iloc[0]
            info = {
                'village': row['Kharif 24 Village'],
                'size': row['Kharif 24 Acres farm/plot'],
                'farm_type': 'TPR' if row['Kharif 24 TPR (Y/N)'] == 1 else 'DSR'
            }

            for idx, meter in enumerate(farms_dict[selected_farm]):
                meter_results.append({
                    'meter': meter,
                    'info': info,
                    'plots': plots64[2*idx:2*idx+2]
                })

    return render(request, 'meter_reading.html', {
        'error_message': error_message,
        'meter_preview': meter_preview,
        'master_previews': master_previews,
        'farm_ids': farm_ids,
        'selected_farm': selected_farm,
        'meter_results': meter_results,
    })


def water_level(request):
    pipe_df = master_df = None
    error_message = None
    pipe_preview = None
    master_previews = {}

    if request.method == 'POST':
        if 'pipe_file' in request.FILES:
            try:
                df = pd.read_excel(request.FILES['pipe_file'])
                pipe_preview = df.head().to_html()
            except Exception as e:
                error_message = f"Error processing pipe file: {e}"

        if 'master_file' in request.FILES:
            try:
                master = pd.read_excel(request.FILES['master_file'], sheet_name=None)
                for name, df in master.items():
                    master_previews[name] = df.head().to_html()
            except Exception as e:
                error_message = f"Error processing master file: {e}"

    return render(request, 'water_level.html', {
        'error_message': error_message,
        'pipe_preview': pipe_preview,
        'master_previews': master_previews,
    })


def farmer_survey(request):
    return render(request, 'farmer_survey.html')


def evapotranspiration(request):
    return render(request, 'evapotranspiration.html')


def mapping(request):
    return render(request, 'mapping.html')


# digivi/views.py

from django.shortcuts import render
import pandas as pd

from .utils import kharif2025_farms, get_2025plots, get_meters_by_village

def meter_reading_25_view(request):
    error = None
    farm_ids = []
    village_names = []  # Add this
    selected = request.POST.get('selected_farm', '')
    selected_village = request.POST.get('selected_village', '')  # Add this
    results = []

    # 1) Upload & cache raw readings
    if request.method == 'POST' and 'raw_file' in request.FILES:
        try:
            raw_df = pd.read_excel(request.FILES['raw_file'])
            request.session['raw25'] = raw_df.to_json(date_format='iso')
        except Exception as e:
            error = f"Error reading raw file: {e}"

    # 2) Upload & cache master workbook
    if request.method == 'POST' and 'meters_file' in request.FILES:
        try:
            master = pd.read_excel(request.FILES['meters_file'], sheet_name=None)
            request.session['master25'] = {
                name: df.to_json(date_format='iso') for name, df in master.items()
            }
        except Exception as e:
            error = f"Error reading master file: {e}"

    # 3) Retrieve from session
    raw_df = pd.read_json(request.session['raw25']) if 'raw25' in request.session else None
    master25 = {
        name: pd.read_json(json_str)
        for name, json_str in request.session.get('master25', {}).items()
    } if 'master25' in request.session else None

    # 4) Build farm dropdown and village dropdown
    if master25:
        farm_dict = kharif2025_farms(master25)
        farm_ids = list(kharif2025_farms(master25).keys())
        
        # Add village extraction
        if raw_df is not None:
            # Column D is index 3 (0-based)
            village_names = sorted(raw_df.iloc[:, 3].dropna().unique().tolist())
        
        download_request = request.POST.get("download_table")
        if download_request and raw_df is not None:
            col_to_get = "m³ per Acre per Avg Day" if download_request == "avg" else "m³ per Acre"
            combined_df = get_tables(raw_df, master25, farm_dict, col_to_get)

            # Convert DataFrame to Excel file in memory
            with BytesIO() as buffer:
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    combined_df.to_excel(writer, index=False, sheet_name='Combined Table')
                buffer.seek(0)
                filename = f"table_{download_request}_kharif2025.xlsx"
                response = HttpResponse(buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                response['Content-Disposition'] = f'attachment; filename="{filename}"'
                return response

    # 5) When a farm is selected, generate graphs
    if selected and raw_df is not None and master25:
        mapping = kharif2025_farms(master25)
        meters = mapping.get(selected, [])
        encoded_imgs = get_2025plots(raw_df, master25, selected, meters)

        # group 4 graphs per meter
        for idx, meter in enumerate(meters):
            block = {
                'meter': meter,
                'plots': encoded_imgs[4*idx : 4*idx + 4]
            }
            results.append(block)
    
    # 6) When a village is selected, generate graphs for all meters in that village
    elif selected_village and raw_df is not None and master25:
        from .utils import get_meters_by_village
        
        # Get all meters for this village
        village_meters = get_meters_by_village(raw_df, selected_village)
        
        # For each meter, find which farm it belongs to and generate graphs
        all_encoded_imgs = []
        meter_to_farm = {}
        
        # Create reverse mapping of meter to farm
        farm_dict = kharif2025_farms(master25)
        for farm_id, meter_list in farm_dict.items():
            for meter in meter_list:
                meter_to_farm[meter] = farm_id
        
        # Generate plots for each meter in the village
        for meter in village_meters:
            if meter in meter_to_farm:
                farm_id = meter_to_farm[meter]
                meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter])
                for idx in range(0, len(meter_imgs), 4):
                    block = {
                        'meter': meter,
                        'farm': farm_id,
                        'plots': meter_imgs[idx:idx+4]
                    }
                    results.append(block)

    return render(request, 'meter_reading_25.html', {
        'error': error,
        'farm_ids': farm_ids,
        'village_names': village_names,  # Add this
        'selected': selected,
        'selected_village': selected_village,  # Add this
        'results': results,
    })


def grouping_25(request):
    if request.method == 'POST':
        selected_label = request.POST.get('group_type')
        selected_checkboxes = request.POST.getlist('group_category')

        raw_df = pd.read_json(request.session['raw25']) if 'raw25' in request.session else None
        master25 = {
            name: pd.read_json(json_str)
            for name, json_str in request.session.get('master25', {}).items()
        } if 'master25' in request.session else None

        kharif_df = master25['Farm details']

        group_farms_dict = {}

        # Map checkbox labels to column names in master file
        group_column_map = {
            "Remote": {
                "Group-A Complied": "Kharif 25 - Remote Controllers Study - Group A - Treatment - complied (Y/N)",
                "Group-A Non-Complied": "Kharif 25 - Remote Controllers Study - Group A - Treatment - NON-complied (Y/N)",
                "Group-B Complied": "Kharif 25 - Remote Controllers Study - Group B - Control - complied (Y/N)",
                "Group-B Non-Complied": "Kharif 25 - Remote Controllers Study - Group B - Control - NON-complied (Y/N)",
            },
            "AWD": {
                "Group-A Complied": "Kharif 25 - AWD Study - Group A - Treatment - complied (Y/N)",
                "Group-A Non-Complied": "Kharif 25 - AWD Study - Group A - Treatment - Non-complied (Y/N)",
                "Group-B Complied": "Kharif 25 - AWD Study - Group B - Complied (Y/N)",
                "Group-B Non-Complied": "Kharif 25 - AWD Study - Group B - Non-complied (Y/N)",
                "Group-C Complied": "Kharif 25 - AWD Study - Group C - Complied (Y/N)",
                "Group-C Non-Complied": "Kharif 25 - AWD Study - Group C - non-complied (Y/N)",
            },
            "TPR/DSR": {
                "TPR": "Kharif 25 - TPR Group Study (Y/N)",
                "DSR": "Kharif 25 - DSR farm Study (Y/N)",
            }
        }



        simplified_groups = {}
        for group in selected_checkboxes:
            base = group.split()[0] 
            label = selected_label + " " + base  
            if label not in simplified_groups:
                simplified_groups[label] = []
            simplified_groups[label].append(group)

        # Build group-wise farm ID lists
        for label, group_list in simplified_groups.items():
            cols = [group_column_map[selected_label][g] for g in group_list]
            condition = (kharif_df[cols[0]].fillna(0) == 1)
            for c in cols[1:]:
                condition |= (kharif_df[c].fillna(0) == 1)
            farm_ids = kharif_df.loc[condition, "Kharif 25 Farm ID"].tolist()
            group_farms_dict[label] = farm_ids

        # Calculate and merge averages
        group_dfs = []
        for label, farms in group_farms_dict.items():
            df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25)
            group_dfs.append(df)

        # Merge all into one plot
        if group_dfs:
            final_df = group_dfs[0]
            for df in group_dfs[1:]:
                final_df = pd.merge(final_df, df, on="Day", how="outer")
            group_plot = generate_group_analysis_plot(final_df)
        
        return render(request, 'grouping.html', {
            'group_plot': group_plot,
            'output': True,
        })
    
    return render(request, 'grouping.html')
            


