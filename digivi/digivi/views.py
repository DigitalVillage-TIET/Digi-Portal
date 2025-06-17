# digivi/views.py

from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
import pandas as pd
from io import BytesIO
from django.http import HttpResponse
from datetime import datetime  # Add this import
from io import BytesIO
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
    village_names = []
    selected = request.POST.get('selected_farm', '')
    selected_village = request.POST.get('selected_village', '')
    results = []
    
    # Date filter variables
    use_date_filter = request.POST.get('use_date_filter') == 'on' or request.session.get('use_date_filter', False)
    filter_start_date = request.POST.get('filter_start_date') or request.session.get('filter_start_date')
    filter_end_date = request.POST.get('filter_end_date') or request.session.get('filter_end_date')

    # 1) Upload & cache raw readings
    if request.method == 'POST' and 'raw_file' in request.FILES:
        try:
            raw_df = pd.read_excel(request.FILES['raw_file'])
            request.session['raw25'] = raw_df.to_json(date_format='iso')
            # Clear date filter when new file is uploaded
            request.session['use_date_filter'] = False
            request.session['filter_start_date'] = None
            request.session['filter_end_date'] = None
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

    # 3) Handle date filter update
    if request.method == 'POST' and 'update_date_filter' in request.POST:
        use_date_filter = request.POST.get('use_date_filter') == 'on'
        request.session['use_date_filter'] = use_date_filter
        
        if use_date_filter:
            filter_start_date = request.POST.get('filter_start_date')
            filter_end_date = request.POST.get('filter_end_date')
            request.session['filter_start_date'] = filter_start_date
            request.session['filter_end_date'] = filter_end_date
        else:
            request.session['filter_start_date'] = None
            request.session['filter_end_date'] = None
        
        # Redirect to same page to refresh with new filter
        return redirect('meter_reading_25')

    # 4) Retrieve from session
    raw_df = pd.read_json(request.session['raw25']) if 'raw25' in request.session else None
    master25 = {
        name: pd.read_json(json_str)
        for name, json_str in request.session.get('master25', {}).items()
    } if 'master25' in request.session else None
    
    # Get date range for display
    date_range_info = None
    if raw_df is not None:
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=False)
        min_date = raw_df['Date'].min().strftime('%Y-%m-%d')
        max_date = raw_df['Date'].max().strftime('%Y-%m-%d')
        date_range_info = {
            'min': min_date,
            'max': max_date,
            'current_min': min_date,
            'current_max': max_date
        }
        
        # Apply date filter if enabled
        if use_date_filter and filter_start_date and filter_end_date:
            from .utils import filter_data_by_date_range
            raw_df = filter_data_by_date_range(raw_df, filter_start_date, filter_end_date)
            date_range_info['current_min'] = filter_start_date
            date_range_info['current_max'] = filter_end_date

    # 5) Build farm dropdown and village dropdown
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
            if use_date_filter and filter_start_date and filter_end_date:
                combined_df = get_tables(raw_df, master25, farm_dict, col_to_get, start_date_enter=filter_start_date, end_date_enter=filter_end_date)
            else:
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

    # Handle Word report download
    if request.method == 'POST' and request.POST.get('download_report') and raw_df is not None and master25:
        # Reconstruct results based on what was selected
        filter_type = None
        filter_value = None
        
        if request.POST.get('report_filter_type') == 'farm':
            selected = request.POST.get('report_filter_value')
            filter_type = "Farm Filter"
            filter_value = selected
            mapping = kharif2025_farms(master25)
            meters = mapping.get(selected, [])
            if use_date_filter and filter_start_date and filter_end_date:
                encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start_date, end_date_enter=filter_end_date)
            else:
                encoded_imgs = get_2025plots(raw_df, master25, selected, meters)
            
            for idx, meter in enumerate(meters):
                block = {
                    'meter': meter,
                    'plots': encoded_imgs[4*idx : 4*idx + 4]
                }
                results.append(block)
                
        elif request.POST.get('report_filter_type') == 'village':
            selected_village = request.POST.get('report_filter_value')
            filter_type = "Village Filter"
            filter_value = selected_village
            from .utils import get_meters_by_village
            
            village_meters = get_meters_by_village(raw_df, selected_village)
            all_encoded_imgs = []
            meter_to_farm = {}
            
            farm_dict = kharif2025_farms(master25)
            for farm_id, meter_list in farm_dict.items():
                for meter in meter_list:
                    meter_to_farm[meter] = farm_id
            
            for meter in village_meters:
                if meter in meter_to_farm:
                    farm_id = meter_to_farm[meter]

                    if use_date_filter and filter_start_date and filter_end_date:
                        meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start_date, end_date_enter=filter_end_date)
                    else:
                        meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter])
                    for idx in range(0, len(meter_imgs), 4):
                        block = {
                            'meter': meter,
                            'farm': farm_id,
                            'plots': meter_imgs[idx:idx+4]
                        }
                        results.append(block)
        
        # Generate Word report
        from .utils import generate_word_report
        docx_buffer = generate_word_report(results, filter_type, filter_value, raw_df, master25)
        
        # Return Word document
        response = HttpResponse(
            docx_buffer.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        response['Content-Disposition'] = f'attachment; filename="water_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.docx"'
        return response

# 6) When a farm is selected, generate graphs
    if selected and raw_df is not None and master25:
        mapping = kharif2025_farms(master25)
        meters = mapping.get(selected, [])
        
        # Generate individual meter plots
        if use_date_filter and filter_start_date and filter_end_date:
            encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start_date, end_date_enter=filter_end_date)
        else:
            encoded_imgs = get_2025plots(raw_df, master25, selected, meters)

        # Group 4 graphs per meter
        for idx, meter in enumerate(meters):
            block = {
                'meter': meter,
                'plots': encoded_imgs[4*idx : 4*idx + 4],
                'is_combined': False
            }
            results.append(block)
        
        # Generate combined plots if multiple meters
        if len(meters) > 1:
            from .utils import get_2025plots_combined
            if use_date_filter and filter_start_date and filter_end_date:
                combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters, start_date_enter=filter_start_date, end_date_enter=filter_end_date)
            else:
                combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters)
            
            if combined_imgs:
                combined_block = {
                    'meter': ' + '.join(meters),
                    'plots': combined_imgs,
                    'is_combined': True,
                    'farm': selected
                }
                results.append(combined_block)
    
    # 7) When a village is selected, generate graphs for all meters in that village
    elif selected_village and raw_df is not None and master25:
        from .utils import get_meters_by_village, get_2025plots_combined
        
        # Get all meters for this village
        village_meters = get_meters_by_village(raw_df, selected_village)
        
        # Create reverse mapping of meter to farm
        farm_dict = kharif2025_farms(master25)
        meter_to_farm = {}
        for farm_id, meter_list in farm_dict.items():
            for meter in meter_list:
                meter_to_farm[meter] = farm_id
        
        # Group meters by farm for combined analysis
        farm_meters_map = {}
        for meter in village_meters:
            if meter in meter_to_farm:
                farm_id = meter_to_farm[meter]
                if farm_id not in farm_meters_map:
                    farm_meters_map[farm_id] = []
                farm_meters_map[farm_id].append(meter)
        
        # Generate plots for each farm in the village
        for farm_id, farm_meters in farm_meters_map.items():
            # Individual meter plots
            for meter in farm_meters:
                if use_date_filter and filter_start_date and filter_end_date:
                    meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start_date, end_date_enter=filter_end_date)
                else:
                    meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter])
                
                for idx in range(0, len(meter_imgs), 4):
                    block = {
                        'meter': meter,
                        'farm': farm_id,
                        'plots': meter_imgs[idx:idx+4],
                        'is_combined': False
                    }
                    results.append(block)
            
            # Combined plots if multiple meters for this farm
            if len(farm_meters) > 1:
                if use_date_filter and filter_start_date and filter_end_date:
                    combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters, start_date_enter=filter_start_date, end_date_enter=filter_end_date)
                else:
                    combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters)
                
                if combined_imgs:
                    combined_block = {
                        'meter': ' + '.join(farm_meters),
                        'farm': farm_id,
                        'plots': combined_imgs,
                        'is_combined': True
                    }
                    results.append(combined_block)

    return render(request, 'meter_reading_25.html', {
        'error': error,
        'farm_ids': farm_ids,
        'village_names': village_names,
        'selected': selected,
        'selected_village': selected_village,
        'results': results,
        'date_range_info': date_range_info,
        'use_date_filter': use_date_filter,
        'filter_start_date': filter_start_date,
        'filter_end_date': filter_end_date,
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
        
        # Apply date filter if enabled
        if raw_df is not None and request.session.get('use_date_filter', False):
            start_date = request.session.get('filter_start_date')
            end_date = request.session.get('filter_end_date')
            if start_date and end_date:
                from .utils import filter_data_by_date_range
                raw_df = filter_data_by_date_range(raw_df, start_date, end_date)

       

        # Handle report download
        if request.POST.get('download_group_report') and 'group_plot' in request.session:
            from .utils import generate_group_analysis_report
            
            # Retrieve stored data from session
            stored_data = request.session.get('group_analysis_data', {})
            
            docx_buffer = generate_group_analysis_report(
                stored_data.get('group_type', ''),
                stored_data.get('selected_groups', []),
                request.session.get('group_plot', ''),
                stored_data.get('group_farms', {})
            )
            
            response = HttpResponse(
                docx_buffer.getvalue(),
                content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
            response['Content-Disposition'] = f'attachment; filename="group_analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.docx"'
            return response

        if selected_label and selected_checkboxes and raw_df is not None and master25:
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
                if raw_df is not None and request.session.get('use_date_filter', False):
                    df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, start_date_enter=start_date, end_date_enter=end_date)
                else:
                    df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25)
                group_dfs.append(df)

            # Merge all into one plot
            if group_dfs:
                final_df = group_dfs[0]
                for df in group_dfs[1:]:
                    final_df = pd.merge(final_df, df, on="Day", how="outer")
                group_plot = generate_group_analysis_plot(final_df)
                
                # Store in session for download
                request.session['group_plot'] = group_plot
                request.session['group_analysis_data'] = {
                    'group_type': selected_label,
                    'selected_groups': selected_checkboxes,
                    'group_farms': group_farms_dict
                }
            
            return render(request, 'grouping.html', {
                'group_plot': group_plot,
                'output': True,
                'group_type': selected_label,
                'selected_groups': selected_checkboxes,
            })
    
    return render(request, 'grouping.html')