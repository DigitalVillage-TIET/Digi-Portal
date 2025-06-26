# digivi/views.py

from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
import pandas as pd
from io import BytesIO
from django.http import HttpResponse, JsonResponse
from datetime import datetime  # Add this import
from io import BytesIO
from .utils import (
    kharif2024_farms, get_2024plots,
    kharif2025_farms, get_2025plots,
    encode_plot_to_base64, get_tables,
    calculate_avg_m3_per_acre, generate_group_analysis_plot
)
from django.urls import reverse
from functools import wraps
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
import jwt
from django.views.decorators.csrf import csrf_exempt
import datetime
from django.conf import settings




# Add these constants at the top of views.py after imports

# Hardcoded credentials for login
JWT_SECRET = 'your_jwt_secret_key'  
JWT_ALGORITHM = 'HS256'
JWT_COOKIE_NAME = 'auth_token'

LOGIN_USERNAME = 'digivi-analyst'
LOGIN_PASSWORD = 'analyst123'

# API credentials (if you're using the API endpoint)
API_USERNAME = 'digivi-analyst'  # Add this line
API_PASSWORD = 'analyst123'      # Add this line

def login_view(request):  # LOGIN VIEW
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            payload = {
                'username': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
                'iat': datetime.datetime.utcnow(),
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
            response = redirect('index')
            response.set_cookie(JWT_COOKIE_NAME, token, httponly=True, samesite='Lax')
            return response
        else:
            error = 'Invalid credentials.'
    
    # If GET request or invalid credentials, show landing page with modal
    return render(request, 'landing.html', {
        'error': error, 
        'show_login_modal': True,
        'is_authenticated': False
    })

def logout_view(request):  # LOGOUT VIEW
    response = redirect('landing')
    response.delete_cookie(JWT_COOKIE_NAME)
    return response

def require_auth(view_func):  
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        token = request.COOKIES.get(JWT_COOKIE_NAME)
        if not token:
            return redirect('landing')
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except Exception:
            return redirect('landing')
        return view_func(request, *args, **kwargs)
    return _wrapped_view

@require_auth
def index(request):
    return render(request, 'index.html')


def landing(request):
    token = request.COOKIES.get(JWT_COOKIE_NAME)
    is_authenticated = False
    if token:
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            is_authenticated = True
        except Exception:
            is_authenticated = False
    
    # If user is already authenticated, redirect to index
    if is_authenticated:
        return redirect('index')
    
    return render(request, 'landing.html', {'is_authenticated': is_authenticated})

@require_auth
def landing_protected(request):
    return render(request, 'landing_protected.html')

@require_auth
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


@require_auth
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


@require_auth
def farmer_survey(request):
    return render(request, 'farmer_survey.html')


@require_auth
def evapotranspiration(request):
    return render(request, 'evapotranspiration.html')


@require_auth
def mapping(request):
    return render(request, 'mapping.html')


# digivi/views.py

from django.shortcuts import render
import pandas as pd

from .utils import kharif2025_farms, get_2025plots, get_meters_by_village

@require_auth
def meter_reading_25_view(request):
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

    # Your Django view code with visual feedback
    error = None
    farm_ids = []
    village_names = []
    selected = request.POST.get('selected_farm', '')
    selected_village = request.POST.get('selected_village', '')
    results = []

    print_status("Initializing view variables", "info")

    # Date filter variables
    use_date_filter = request.POST.get('use_date_filter') == 'on' or request.session.get('use_date_filter', False)
    filter_start_date = request.POST.get('filter_start_date') or request.session.get('filter_start_date')
    filter_end_date = request.POST.get('filter_end_date') or request.session.get('filter_end_date')

    if use_date_filter:
        print_status(f"Date filter enabled: {filter_start_date} to {filter_end_date}", "info")
    else:
        print_status("Date filter disabled", "info")

    # 1) Upload & cache raw readings + Load master from Google Sheets
    if request.method == 'POST' and 'raw_file' in request.FILES:
        try:
            print_status("Processing raw file upload...", "process")
            
            # Load raw file
            print_status("Reading Excel file...", "process")
            raw_df = pd.read_excel(request.FILES['raw_file'])
            print_status(f"Raw file loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns", "success")
            
            # Clean raw data to handle empty entries
            print_status("Cleaning raw data...", "process")
            raw_df_cleaned = raw_df.replace(r'^\s*$', pd.NA, regex=True)  # Empty or whitespace-only
            raw_df_cleaned = raw_df_cleaned.replace('', pd.NA)  # Explicit empty strings
            raw_df_cleaned = raw_df_cleaned.replace('nan', pd.NA)  # String 'nan'
            raw_df_cleaned = raw_df_cleaned.replace('NaN', pd.NA)  # String 'NaN'
            print_status("Empty values cleaned", "success")
            
            # Convert columns to appropriate data types where possible
            print_status("Converting data types...", "process")
            numeric_cols = 0
            for col in raw_df_cleaned.columns:
                original_type = raw_df_cleaned[col].dtype
                raw_df_cleaned[col] = pd.to_numeric(raw_df_cleaned[col], errors='ignore')
                if raw_df_cleaned[col].dtype != original_type and pd.api.types.is_numeric_dtype(raw_df_cleaned[col]):
                    numeric_cols += 1
            
            if numeric_cols > 0:
                print_status(f"Converted {numeric_cols} columns to numeric types", "success")
            else:
                print_status("No columns converted to numeric", "info")
            
            print_status("Caching raw data in session...", "process")
            request.session['raw25'] = raw_df_cleaned.to_json(date_format='iso')
            print_status("Raw data cached successfully", "success")
        
            # Load master data from Google Sheets
            print_status("Loading master data from Google Sheets...", "process")
            from .utils import load_master_from_google_sheets
            master25 = load_master_from_google_sheets()
            
            if master25:
                print_status(f"Master data loaded from Google Sheets: {len(master25)} sheets", "success")
                
                # Clean and process master data to handle empty entries
                print_status("Processing master data sheets...", "process")
                cleaned_master = {}
                
                for name, df in master25.items():
                    print_status(f"Cleaning sheet: {name}", "process")
                    
                    # Replace empty strings, whitespace-only strings, and 'nan' strings with actual NaN
                    df_cleaned = df.replace(r'^\s*$', pd.NA, regex=True)  # Empty or whitespace-only
                    df_cleaned = df_cleaned.replace('', pd.NA)  # Explicit empty strings
                    df_cleaned = df_cleaned.replace('nan', pd.NA)  # String 'nan'
                    df_cleaned = df_cleaned.replace('NaN', pd.NA)  # String 'NaN'
                    
                    # Convert columns to appropriate data types where possible
                    # This helps ensure numeric columns don't have string representations of numbers
                    numeric_conversions = 0
                    for col in df_cleaned.columns:
                        original_type = df_cleaned[col].dtype
                        # Try to convert to numeric, keeping NaN for non-numeric values
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
                        if df_cleaned[col].dtype != original_type and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                            numeric_conversions += 1
                    
                    cleaned_master[name] = df_cleaned
                    print_status(f"Sheet '{name}' processed: {len(df_cleaned)} rows, {numeric_conversions} numeric columns", "success")
                
                print_status("Caching master data in session...", "process")
                request.session['master25'] = {
                    name: df.to_json(date_format='iso') for name, df in cleaned_master.items()
                }
                print_status("Master data cached successfully", "success")
            else:
                error = "Failed to load master data from Google Sheets"
                print_status(error, "error")
        
            # Clear date filter when new file is uploaded
            print_status("Clearing previous date filters...", "process")
            request.session['use_date_filter'] = False
            request.session['filter_start_date'] = None
            request.session['filter_end_date'] = None
            print_status("Date filters cleared", "success")
            
            print_status("File upload and processing completed successfully", "success")
            
        except Exception as e:
            error = f"Error processing files: {e}"
            print_status(error, "error")

    # 2) Manual master workbook upload (fallback option)
    if request.method == 'POST' and 'meters_file' in request.FILES:
        try:
            print_status("Processing manual master workbook upload...", "process")
            
            print_status("Reading Excel workbook with all sheets...", "process")
            master = pd.read_excel(request.FILES['meters_file'], sheet_name=None)
            print_status(f"Master workbook loaded: {len(master)} sheets", "success")
            
            # Apply same cleaning logic to manual uploads
            print_status("Cleaning master workbook data...", "process")
            cleaned_master = {}
            
            for name, df in master.items():
                print_status(f"Processing sheet: {name}", "process")
                
                # Replace empty strings, whitespace-only strings, and 'nan' strings with actual NaN
                df_cleaned = df.replace(r'^\s*$', pd.NA, regex=True)  # Empty or whitespace-only
                df_cleaned = df_cleaned.replace('', pd.NA)  # Explicit empty strings
                df_cleaned = df_cleaned.replace('nan', pd.NA)  # String 'nan'
                df_cleaned = df_cleaned.replace('NaN', pd.NA)  # String 'NaN'
                
                # Convert columns to appropriate data types where possible
                numeric_conversions = 0
                for col in df_cleaned.columns:
                    original_type = df_cleaned[col].dtype
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
                    if df_cleaned[col].dtype != original_type and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        numeric_conversions += 1
                
                cleaned_master[name] = df_cleaned
                print_status(f"Sheet '{name}' cleaned: {len(df_cleaned)} rows, {numeric_conversions} numeric columns", "success")
            
            print_status("Caching cleaned master data in session...", "process")
            request.session['master25'] = {
                name: df.to_json(date_format='iso') for name, df in cleaned_master.items()
            }
            print_status("Manual master workbook processed and cached successfully", "success")
            
        except Exception as e:
            error = f"Error reading master file: {e}"
            print_status(error, "error")

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
                filter_start = pd.to_datetime(filter_start_date)
                filter_end = pd.to_datetime(filter_end_date)
                combined_df = get_tables(raw_df, master25, farm_dict, col_to_get, start_date_enter=filter_start, end_date_enter=filter_end)
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
                filter_start = pd.to_datetime(filter_start_date)
                filter_end = pd.to_datetime(filter_end_date)
                encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
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
                        filter_start = pd.to_datetime(filter_start_date)
                        filter_end = pd.to_datetime(filter_end_date)
                        meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
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
            filter_start = pd.to_datetime(filter_start_date)
            filter_end = pd.to_datetime(filter_end_date)
            encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
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
                filter_start = pd.to_datetime(filter_start_date)
                filter_end = pd.to_datetime(filter_end_date)
                combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
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
                    filter_start = pd.to_datetime(filter_start_date)
                    filter_end = pd.to_datetime(filter_end_date)
                    meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
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
                    filter_start = pd.to_datetime(filter_start_date)
                    filter_end = pd.to_datetime(filter_end_date)
                    combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters, start_date_enter=filter_start, end_date_enter=filter_end)
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

@require_auth
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
                request.session.get('group_plot2', ''),
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
            group_dfs2 = []
            for label, farms in group_farms_dict.items():
                if raw_df is not None and request.session.get('use_date_filter', False):
                    filter_start = pd.to_datetime(start_date)
                    filter_end = pd.to_datetime(end_date)
                    df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, 'm³ per Acre per Avg Day' ,start_date_enter=filter_start, end_date_enter=filter_end)
                    df2 = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, "Delta m³" ,start_date_enter=filter_start, end_date_enter=filter_end)
                else:
                    df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, 'm³ per Acre per Avg Day')
                    df2 = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, "Delta m³")
                group_dfs.append(df)
                group_dfs2.append(df2)

            # Merge all into one plot
            if group_dfs:
                final_df = group_dfs[0]
                for df in group_dfs[1:]:
                    final_df = pd.merge(final_df, df, on="Day", how="outer")
                group_plot = generate_group_analysis_plot(final_df, "Daily Average m3/acre")
                
                # Store in session for download
                request.session['group_plot'] = group_plot
                request.session['group_analysis_data'] = {
                    'group_type': selected_label,
                    'selected_groups': selected_checkboxes,
                    'group_farms': group_farms_dict
                }
            
            group_plot2 = None
            if group_dfs2:
                final_df2 = group_dfs2[0]
                for df in group_dfs2[1:]:
                    final_df2 = pd.merge(final_df2, df, on="Day", how="outer")
                group_plot2 = generate_group_analysis_plot(final_df2, "Delta m3/acre")
                
                # Store in session for download
                request.session['group_plot2'] = group_plot2
               
            return render(request, 'grouping.html', {
                'group_plot': group_plot,
                'group_plot2': group_plot2,
                'output': True,
                'group_type': selected_label,
                'selected_groups': selected_checkboxes,
            })
    
    return render(request, 'grouping.html')

@csrf_exempt
def api_token(request):
    if request.method == 'POST':
        import json
        data = json.loads(request.body.decode())
        username = data.get('username')
        password = data.get('password')
        if username == API_USERNAME and password == API_PASSWORD:
            payload = {
                'username': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
                'iat': datetime.datetime.utcnow(),
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
            return JsonResponse({'access': token})
        else:
            return JsonResponse({'detail': 'Invalid credentials'}, status=401)
    return JsonResponse({'detail': 'Method not allowed'}, status=405)

    