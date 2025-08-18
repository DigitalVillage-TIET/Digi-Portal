# # digivi/views.py

# from django.shortcuts import render, redirect
# from django.core.files.storage import default_storage
# import pandas as pd
# from io import BytesIO, StringIO
# from django.http import HttpResponse, JsonResponse
# import datetime  # Add this import
# import io
# import zipfile
# import re
# import jwt
# # In views.py, update the imports section:
# from .utils import (
#     kharif2024_farms, get_2024plots,
#     kharif2025_farms, get_2025plots, get_2025plots_combined,
#     get_2025plots_plotly, get_2025plots_combined_plotly, 
#     encode_plot_to_base64, get_tables,
#     calculate_avg_m3_per_acre, generate_group_analysis_plot,
#     load_and_validate_data,
#     clean_and_process_data,
#     create_merged_dataset,
#     apply_comprehensive_filters,
#     render_comparative_analysis,
#     get_farm_analysis_data,
#     create_zip_package,
#     prepare_comprehensive_downloads,
#     get_per_farm_downloads,
#     get_village_level_analysis,
#     load_master_from_google_sheets,
#     print_status,
#     generate_delta_vs_days_groupwise_plots
# )
# from django.urls import reverse
# from functools import wraps
# from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
# import jwt
# from django.views.decorators.csrf import csrf_exempt
# from django.conf import settings




# # Add these constants at the top of views.py after imports

# # Hardcoded credentials for login
# JWT_SECRET = 'your_jwt_secret_key'  
# JWT_ALGORITHM = 'HS256'
# JWT_COOKIE_NAME = 'auth_token'

# LOGIN_USERNAME = 'digivi-analyst'
# LOGIN_PASSWORD = 'analyst123'

# # API credentials (if you're using the API endpoint)
# API_USERNAME = 'digivi-analyst'  # Add this line
# API_PASSWORD = 'analyst123'      # Add this line

# def login_view(request):  # LOGIN VIEW
#     error = None
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
#             payload = {
#                 'username': username,
#                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
#                 'iat': datetime.datetime.utcnow(),
#             }
#             token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
#             response = redirect('index')
#             response.set_cookie(JWT_COOKIE_NAME, token, httponly=True, samesite='Lax')
#             return response
#         else:
#             error = 'Invalid credentials.'
    
#     # If GET request or invalid credentials, show landing page with modal
#     return render(request, 'landing.html', {
#         'error': error, 
#         'show_login_modal': True,
#         'is_authenticated': False
#     })

# def logout_view(request):  # LOGOUT VIEW
#     response = redirect('landing')
#     response.delete_cookie(JWT_COOKIE_NAME)
#     return response

# def require_auth(view_func):  
#     @wraps(view_func)
#     def _wrapped_view(request, *args, **kwargs):
#         token = request.COOKIES.get(JWT_COOKIE_NAME)
#         if not token:
#             return redirect('landing')
#         try:
#             jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#         except Exception:
#             return redirect('landing')
#         return view_func(request, *args, **kwargs)
#     return _wrapped_view

# @require_auth
# def index(request):
#     return render(request, 'index.html')


# def landing(request):
#     token = request.COOKIES.get(JWT_COOKIE_NAME)
#     is_authenticated = False
#     if token:
#         try:
#             jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#             is_authenticated = True
#         except Exception:
#             is_authenticated = False
    
#     # If user is already authenticated, redirect to index
#     if is_authenticated:
#         return redirect('index')
    
#     return render(request, 'landing.html', {'is_authenticated': is_authenticated})

# @require_auth
# def landing_protected(request):
#     return render(request, 'landing_protected.html')

# @require_auth
# def meter_reading(request):
#     meter_df = master_df = None
#     error_message = meter_preview = None
#     master_previews = {}
#     farm_ids = []
#     meter_results = []
#     selected_farm = request.POST.get('selected_farm', "")

#     if request.method == 'POST':
#         # Upload & cache meter file
#         if 'meter_file' in request.FILES:
#             try:
#                 meter_df = pd.read_excel(request.FILES['meter_file'])
#                 date_col = meter_df.columns[0]
#                 meter_df[date_col] = pd.to_datetime(meter_df[date_col], errors='coerce')
#                 request.session['meter_df'] = meter_df.to_json(date_format='iso')
#                 meter_preview = meter_df.head().to_html()
#             except Exception as e:
#                 error_message = f"Error processing meter file: {e}"

#         # Upload & cache master file
#         if 'master_file' in request.FILES:
#             try:
#                 master = pd.read_excel(request.FILES['master_file'], sheet_name=None)
#                 request.session['master_df'] = {
#                     k: v.to_json(date_format='iso') for k, v in master.items()
#                 }
#                 for name, df in master.items():
#                     master_previews[name] = df.head().to_html()
#             except Exception as e:
#                 error_message = f"Error processing master file: {e}"

#         # Retrieve from session if missing
#         if meter_df is None and 'meter_df' in request.session:
#             meter_df = pd.read_json(request.session['meter_df'])
#         if master_df is None and 'master_df' in request.session:
#             master_df = {k: pd.read_json(v) for k, v in request.session['master_df'].items()}
#             for name, df in master_df.items():
#                 master_previews[name] = df.head().to_html()

#         # Farm dropdown
#         if master_df:
#             farms_dict = kharif2024_farms(master_df)
#             farm_ids = list(farms_dict.keys())

#         # Generate plots
#         if selected_farm and master_df:
#             figs = get_2024plots(selected_farm, meter_df, master_df, farms_dict[selected_farm])
#             plots64 = [encode_plot_to_base64(f) for f in figs]

#             kharif = master_df['Kharif 24']
#             row = kharif[kharif['Kharif 24 FarmID'] == selected_farm].iloc[0]
#             info = {
#                 'village': row['Kharif 24 Village'],
#                 'size': row['Kharif 24 Acres farm/plot'],
#                 'farm_type': 'TPR' if row['Kharif 24 TPR (Y/N)'] == 1 else 'DSR'
#             }

#             for idx, meter in enumerate(farms_dict[selected_farm]):
#                 meter_results.append({
#                     'meter': meter,
#                     'info': info,
#                     'plots': plots64[2*idx:2*idx+2]
#                 })

#     return render(request, 'meter_reading.html', {
#         'error_message': error_message,
#         'meter_preview': meter_preview,
#         'master_previews': master_previews,
#         'farm_ids': farm_ids,
#         'selected_farm': selected_farm,
#         'meter_results': meter_results,
#     })

# def get_comparative_analysis(section, merged_df, kharif_df, custom_groups_df):
#     comparative = render_comparative_analysis(merged_df, kharif_df, custom_groups_df)
#     if section == 'village':
#         return comparative['village']['summary_df']
#     elif section == 'rc':
#         return comparative['rc']['summary_df']
#     elif section == 'awd':
#         return comparative['awd']['summary_df']
#     elif section == 'dsr_tpr':
#         return comparative['dsr_tpr']['summary_df']
#     elif section == 'custom':
#         return comparative['custom']['summary_df']
#     else:
#         return None
    

# def sanitize_filename(filename):
#     """Sanitize filename for safe file operations"""
#     return re.sub(r'[^\w\-_\.]', '_', str(filename))

# def load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session):
#     """Load all necessary context to prevent page breaking when errors occur"""
#     def categorize_files(downloadables):
#         complete_dataset = [f for f in downloadables if f.startswith('01_') or f.startswith('02_')]
#         study_group = [f for f in downloadables if any(k in f.lower() for k in ['remote', 'awd', 'dsr', 'tpr'])]
#         analysis_reports = [f for f in downloadables if 'village_summary' in f or 'farm_summary' in f]
#         return complete_dataset, study_group, analysis_reports
 

# from io import StringIO
# from django.shortcuts import render
# from django.http import HttpResponse
# from .utils import (
#     load_and_validate_data,
#     clean_and_process_data,
#     create_merged_dataset,
#     apply_comprehensive_filters,
#     render_comparative_analysis,
#     get_farm_analysis_data,
#     create_zip_package,
#     prepare_comprehensive_downloads,
#     get_per_farm_downloads,
#     get_village_level_analysis
# )
# import pandas as pd
# import json
# import io
# import zipfile
# from io import StringIO
# import re
# def water_dashboard_view(request):
#     context = {}
#     session = request.session

#     def categorize_files(downloadables):
#         complete_dataset = [f for f in downloadables if f.startswith('01_') or f.startswith('02_')]
#         study_group = [f for f in downloadables if any(k in f.lower() for k in ['remote', 'awd', 'dsr', 'tpr'])]
#         analysis_reports = [f for f in downloadables if 'village_summary' in f or 'farm_summary' in f]
#         return complete_dataset, study_group, analysis_reports

#     # ----------------------------------
#     # 0️⃣ Handle Direct CSV Download
#     # ----------------------------------
#     if 'download' in request.GET:
#         filename = request.GET['download']
#         if 'merged_df' not in session or 'kharif_df' not in session:
#             return HttpResponse(b"No data loaded.", status=400)

#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         kharif_df = pd.read_json(session['kharif_df'], orient='split')
#         farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split') if 'farm_daily_avg' in session else None
#         weekly_avg = pd.read_json(session['weekly_avg'], orient='split') if 'weekly_avg' in session else None

#         downloadables = prepare_comprehensive_downloads(merged_df, kharif_df, farm_daily_avg, weekly_avg)
#         if filename.startswith('farm_'):
#             downloadables.update(get_per_farm_downloads(merged_df, farm_daily_avg))
#         elif filename.startswith('village_'):
#             downloadables.update(get_village_level_analysis(merged_df, kharif_df))
#         if filename not in downloadables:
#             return HttpResponse(b"File not found.", status=404)

#         response = HttpResponse(downloadables[filename], content_type='text/csv')
#         response['Content-Disposition'] = f'attachment; filename={filename}'
#         return response

#     # ----------------------------------
#     # 1️⃣ Handle File Upload
#     # ----------------------------------
#     if request.method == 'POST' and 'kharif_file' in request.FILES and 'water_file' in request.FILES:
#         kharif_df, water_df, missing_kharif, missing_water = load_and_validate_data(
#             request.FILES['kharif_file'], request.FILES['water_file']
#         )
#         if missing_kharif or missing_water:
#             context['error'] = f"Missing columns: Kharif: {missing_kharif}, Water: {missing_water}"
#             return render(request, 'water_meters.html', context)

#         kharif_cleaned, water_cleaned = clean_and_process_data(kharif_df, water_df)
#         filtered_kharif, filtered_water = apply_comprehensive_filters(kharif_cleaned, water_cleaned, {
#             'villages': [], 'date_range': None, 'available_groups': {},
#             'remote_controllers': 'All', 'awd_study': 'All',
#             'farming_method': 'All', 'min_readings': 1, 'remove_outliers': False
#         })

#         merged_df, farm_daily_avg, weekly_avg = create_merged_dataset(filtered_kharif, filtered_water)

#         session['merged_df'] = merged_df.to_json(orient='split')
#         session['kharif_df'] = kharif_cleaned.to_json(orient='split')
#         session['filtered_kharif'] = filtered_kharif.to_json(orient='split')
#         session['filtered_water'] = filtered_water.to_json(orient='split')
#         session['farm_daily_avg'] = farm_daily_avg.to_json(orient='split')
#         session['weekly_avg'] = weekly_avg.to_json(orient='split')

#         context['summary'] = {
#             'total_records': len(merged_df),
#             'unique_farms': merged_df['Farm ID'].nunique(),
#             'total_villages': merged_df['Village'].nunique() if 'Village' in merged_df.columns else 0,
#             'date_range': f"{merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}",
#             'avg_level': round(merged_df['Water Level (mm)'].mean(), 1)
#         }
#         context['merged_loaded'] = True
#         context['available_farms'] = sorted(merged_df["Farm ID"].unique())
#         context['available_villages'] = sorted(merged_df["Village"].dropna().unique()) if 'Village' in merged_df.columns else []
#         context['selected_farm'] = context['available_farms'][0] if context['available_farms'] else None
#         farm = context['selected_farm']
#         context['per_farm_downloads'] = [
#     {
#         'filename': f"farm_{sanitize_filename(farm)}_{suffix}.csv",
#         'display_name': label
#     }
#     for suffix, label in zip(
#         ['detailed', 'summary', 'daily_avg'],
#         [("Detailed Data"), ("Summary Row"), ("Daily Averages")]
#     )
#     ] if farm else []
#         # 🔍 Load Valid Custom Groups
#         custom_groups_df = {}
#         for k, v in session.get('custom_groups', {}).items():
#             try:
#                 df = pd.read_json(StringIO(v), orient='split')
#                 if df.empty:
#                     print(f"[WARN] Custom group '{k}' is empty.")
#                     continue
#                 required = {"Farm ID", "Village", "Water Level (mm)", "Date", "Days from TPR"}
#                 if not required.issubset(df.columns):
#                     print(f"[WARN] Custom group '{k}' missing columns: {required - set(df.columns)}")
#                     continue
#                 custom_groups_df[k] = df
#             except Exception as e:
#                 print(f"[ERROR] Failed to parse custom group '{k}':", e)

#         print("[DEBUG] Valid custom group keys:", list(custom_groups_df.keys()))

#         # 📊 Comparative Analysis
#         comparative = render_comparative_analysis(merged_df, kharif_cleaned, custom_groups_df)
#         context['comparative'] = {
#             'village': {
#                 'plot': comparative['village']['plot'].to_html(full_html=False),
#                 'summary_df': comparative['village']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'rc': {
#                 'plot': comparative['rc']['plot'].to_html(full_html=False),
#                 'summary_df': comparative['rc']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'awd': {
#                 'plot': comparative['awd']['plot'].to_html(full_html=False),
#                 'summary_df': comparative['awd']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'dsr_tpr': {
#                 'plot': comparative['dsr_tpr']['plot'].to_html(full_html=False),
#                 'summary_df': comparative['dsr_tpr']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'compliance': {
#                 'df': comparative['compliance']['df'].to_html(index=False, classes="table table-striped"),
#                 'summary': comparative['compliance']['summary']
#             },
#             'custom': {
#                 'summary_df': (
#                     comparative['custom']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#                     if comparative['custom'].get('summary_df') is not None and not comparative['custom']['summary_df'].empty
#                     else "<div class='text-muted'>No valid custom groups available.</div>"
#                 ),
#                 'chart_html': comparative['custom'].get('chart_html', None)
#             }
#         }
#         context['custom_groups'] = list(custom_groups_df.keys())

#         # 📥 Download Links
#         downloads = prepare_comprehensive_downloads(merged_df, kharif_cleaned, farm_daily_avg, weekly_avg)
#         ds1, ds2, ds3 = categorize_files(downloads)
#         context['complete_dataset_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds1]
#         context['study_group_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds2]
#         context['analysis_report_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds3]

#         # 📊 Overall Stats
#         stats = {
#             'mean': round(merged_df['Water Level (mm)'].mean(), 2),
#             'std': round(merged_df['Water Level (mm)'].std(), 2),
#             'min': round(merged_df['Water Level (mm)'].min(), 2),
#             'max': round(merged_df['Water Level (mm)'].max(), 2),
#             'total_readings': len(merged_df),
#             'unique_farms': merged_df['Farm ID'].nunique()
#         }
#         context['farm_summary_stats'] = stats

#         # Selected farm downloads
#         farm = context['selected_farm']
#         context['per_farm_downloads'] = [
#             {'filename': f"farm_{sanitize_filename(farm)}_{suffix}.csv", 'display_name': label}
#             for suffix, label in zip(['detailed', 'summary', 'daily_avg'], ["Detailed Data", "Summary Row", "Daily Averages"])
#         ] if farm else []

#     # ----------------------------------
#     # 2️⃣ Handle AJAX Farm Selection
#     # ----------------------------------
#     if 'merged_df' in session and request.method == 'POST' and 'download_zip' not in request.POST:
#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         kharif_df = pd.read_json(session['kharif_df'], orient='split')
#         farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
#         weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
#         available_farms = sorted(merged_df["Farm ID"].unique())
#         selected_farm = request.POST.get('selected_farm') or (available_farms[0] if available_farms else None)
#         context.update({
#             'available_farms': available_farms,
#             'selected_farm': selected_farm,
#             'merged_loaded': True
#         })
#         if selected_farm:
#             analysis = get_farm_analysis_data(selected_farm, merged_df, farm_daily_avg, weekly_avg)
#             context['farm_analysis'] = {
#                 "plot1": analysis["plot1"],
#                 "plot2": analysis["plot2"],
#                 "summary": analysis["summary"],
#                 "pivot_table_html": analysis["pivot_table"]
#             }
#             context['farm_summary_stats'] = {
#                 'mean': round(analysis["summary"]["average"], 1),
#                 'std': round(analysis["summary"]["std_dev"], 1),
#                 'min': round(analysis["summary"]["min"], 1),
#                 'max': round(analysis["summary"]["max"], 1),
#             }
#             context['per_farm_downloads'] = [
#                 {'filename': f"farm_{sanitize_filename(selected_farm)}_{suffix}.csv", 'display_name': label}
#                 for suffix, label in zip(['detailed', 'summary', 'daily_avg'], ["Detailed Data", "Summary Row", "Daily Averages"])
#             ]
#         if 'farm_select' in request.POST:
#             return render(request, 'water_meters.html', context)

#     # ----------------------------------
#     # 3️⃣ Download Full ZIP
#     # ----------------------------------
#     if request.method == 'POST' and 'download_zip' in request.POST:
#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         kharif_df = pd.read_json(session['kharif_df'], orient='split')
#         farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
#         weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
#         zip_buffer = create_zip_package(merged_df, kharif_df, farm_daily_avg, weekly_avg)
#         response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
#         response['Content-Disposition'] = 'attachment; filename=agri_dashboard_package.zip'
#         return response

#     # ----------------------------------
#     # 4️⃣ Village Summary Download
#     # ----------------------------------
#     if 'download_village_summary' in request.GET:
#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         selected_villages = request.GET.getlist('villages')
#         filtered_df = merged_df[merged_df['Village'].isin(selected_villages)] if selected_villages else merged_df
#         summary_csv = get_village_level_analysis(filtered_df)['summary_df'].to_csv(index=False)
#         return HttpResponse(summary_csv, content_type='text/csv', headers={
#             'Content-Disposition': 'attachment; filename=village_summary.csv'
#         })

#     # ----------------------------------
#     # 5️⃣ Village Analysis ZIP
#     # ----------------------------------
#     if 'download_village_analysis' in request.GET:
#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         selected_villages = request.GET.getlist('villages')
#         filtered_df = merged_df[merged_df['Village'].isin(selected_villages)] if selected_villages else merged_df
#         village_result = get_village_level_analysis(filtered_df)

#         zip_buffer = io.BytesIO()
#         with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
#             zf.writestr('village_summary.csv', village_result['summary_df'].to_csv(index=False))
#             if 'trend_df' in village_result:
#                 zf.writestr('village_daily_trends.csv', village_result['trend_df'].to_csv(index=False))
#         zip_buffer.seek(0)
#         return HttpResponse(zip_buffer.getvalue(), content_type='application/zip', headers={
#             'Content-Disposition': 'attachment; filename=village_analysis.zip'
#         })

#     # ----------------------------------
#     # 6️⃣ Handle Custom Group Creation
#     # ----------------------------------
#     if request.method == 'POST' and 'create_custom_group' in request.POST:
#         if 'merged_df' not in session:
#             context['error'] = "No data loaded. Please upload files first."
#             return render(request, 'water_meters.html', context)

#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         kharif_df = pd.read_json(session['kharif_df'], orient='split')
#         farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
#         weekly_avg = pd.read_json(session['weekly_avg'], orient='split')

#         group_count = int(request.POST.get('group_count', 1))
#         custom_groups = session.get('custom_groups', {})

#         group_created = False
#         for i in range(1, group_count + 1):
#             group_name = request.POST.get(f'group_name_{i}', '').strip()
#             selected_farms = request.POST.getlist(f'selected_farms_{i}')
#             selected_villages = request.POST.getlist(f'selected_villages_{i}')
#             if not group_name:
#                 context['error'] = f"Group name is required for group {i}."
#                 return render(request, 'water_meters.html', context)
#             if not selected_farms and not selected_villages:
#                 context['error'] = f"Please select at least one farm or village for group {i}."
#                 return render(request, 'water_meters.html', context)
#             # Create custom group DataFrame
#             custom_group_data = merged_df.copy()
#             if selected_farms:
#                 custom_group_data = custom_group_data[custom_group_data['Farm ID'].isin(selected_farms)]
#             if selected_villages:
#                 custom_group_data = custom_group_data[custom_group_data['Village'].isin(selected_villages)]
#             if custom_group_data.empty:
#                 context['error'] = f"No data found for the selected farms/villages in group {i}."
#                 return render(request, 'water_meters.html', context)
#             custom_groups[group_name] = custom_group_data.to_json(orient='split')
#             group_created = True

#         if group_created:
#             session['custom_groups'] = custom_groups
#             session.modified = True
#             context['success_message'] = f"Custom group(s) created successfully."

#         # Regenerate comparative analysis with updated custom groups
#         custom_groups_df = {k: pd.read_json(StringIO(v), orient='split') for k, v in custom_groups.items()}
#         comparative_results = render_comparative_analysis(merged_df, kharif_df, custom_groups_df)
        
#         # Set up context for comparative analysis
#         context['comparative'] = {
#             'village': {
#                 'plot': comparative_results['village']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['village']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'rc': {
#                 'plot': comparative_results['rc']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['rc']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'awd': {
#                 'plot': comparative_results['awd']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['awd']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'dsr_tpr': {
#                 'plot': comparative_results['dsr_tpr']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['dsr_tpr']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'compliance': {
#                 'df': comparative_results['compliance']['df'].to_html(index=False, classes="table table-striped"),
#                 'summary': comparative_results['compliance']['summary']
#             },
#             'custom': {
#                 'summary_df': (
#                     comparative_results['custom']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#                     if comparative_results['custom'].get('summary_df') is not None and not comparative_results['custom']['summary_df'].empty
#                     else "<div class='text-muted'>No valid custom groups available.</div>"
#                 ),
#                 'chart_html': comparative_results['custom'].get('chart_html', None)
#             }
#         }
#         context['custom_groups'] = list(custom_groups_df.keys())

#         # 📥 Download Links
#         downloads = prepare_comprehensive_downloads(merged_df, kharif_df, farm_daily_avg, weekly_avg)
#         ds1, ds2, ds3 = categorize_files(downloads)
#         context['complete_dataset_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds1]
#         context['study_group_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds2]
#         context['analysis_report_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds3]

#         # 📊 Overall Stats
#         stats = {
#             'mean': round(merged_df['Water Level (mm)'].mean(), 2),
#             'std': round(merged_df['Water Level (mm)'].std(), 2),
#             'min': round(merged_df['Water Level (mm)'].min(), 2),
#             'max': round(merged_df['Water Level (mm)'].max(), 2),
#             'total_readings': len(merged_df),
#             'unique_farms': merged_df['Farm ID'].nunique()
#         }
#         context['farm_summary_stats'] = stats

#         # Selected farm downloads
#         farm = context['selected_farm']
#         context['per_farm_downloads'] = [
#             {'filename': f"farm_{sanitize_filename(farm)}_{suffix}.csv", 'display_name': label}
#             for suffix, label in zip(['detailed', 'summary', 'daily_avg'], ["Detailed Data", "Summary Row", "Daily Averages"])
#         ] if farm else []

#     # ----------------------------------
#     # 7️⃣ Handle Custom Group Deletion
#     # ----------------------------------
#     if request.method == 'POST' and 'delete_custom_group' in request.POST:
#         if 'merged_df' not in session:
#             context['error'] = "No data loaded. Please upload files first."
#             return render(request, 'water_meters.html', context)
        
#         merged_df = pd.read_json(session['merged_df'], orient='split')
#         kharif_df = pd.read_json(session['kharif_df'], orient='split')
#         farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
#         weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
#         group_name = request.POST.get('group_name', '').strip()
#         if not group_name:
#             context['error'] = "Group name is required for deletion."
#             # Load context to prevent breaking
#             context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
#             return render(request, 'water_meters.html', context)
        
#         custom_groups = session.get('custom_groups', {})
#         if group_name in custom_groups:
#             del custom_groups[group_name]
#             session['custom_groups'] = custom_groups
#             session.modified = True
#             context['success_message'] = f"Custom group '{group_name}' deleted successfully."
#         else:
#             context['error'] = f"Custom group '{group_name}' not found."
#             # Load context to prevent breaking
#             context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
#             return render(request, 'water_meters.html', context)
        
#         # Regenerate comparative analysis with updated custom groups
#         custom_groups_df = {k: pd.read_json(StringIO(v), orient='split') for k, v in custom_groups.items()}
#         comparative_results = render_comparative_analysis(merged_df, kharif_df, custom_groups_df)
        
#         # Set up context for comparative analysis
#         context['comparative'] = {
#             'village': {
#                 'plot': comparative_results['village']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['village']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'rc': {
#                 'plot': comparative_results['rc']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['rc']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'awd': {
#                 'plot': comparative_results['awd']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['awd']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'dsr_tpr': {
#                 'plot': comparative_results['dsr_tpr']['plot'].to_html(full_html=False),
#                 'summary_df': comparative_results['dsr_tpr']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#             },
#             'compliance': {
#                 'df': comparative_results['compliance']['df'].to_html(index=False, classes="table table-striped"),
#                 'summary': comparative_results['compliance']['summary']
#             },
#             'custom': {
#                 'summary_df': (
#                     comparative_results['custom']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
#                     if comparative_results['custom'].get('summary_df') is not None and not comparative_results['custom']['summary_df'].empty
#                     else "<div class='text-muted'>No valid custom groups available.</div>"
#                 ),
#                 'chart_html': comparative_results['custom'].get('chart_html', None)
#             }
#         }
        
#         # Set up other required context
#         context['merged_loaded'] = True
#         context['available_farms'] = sorted(merged_df["Farm ID"].unique())
#         context['available_villages'] = sorted(merged_df["Village"].dropna().unique()) if 'Village' in merged_df.columns else []
#         context['custom_groups'] = list(custom_groups_df.keys())
        
#         # Set up download categories
#         downloads = prepare_comprehensive_downloads(merged_df, kharif_df, farm_daily_avg, weekly_avg)
#         ds1, ds2, ds3 = categorize_files(downloads)
#         context['complete_dataset_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds1]
#         context['study_group_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds2]
#         context['analysis_report_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds3]
        
#         # Farm summary statistics
#         stats = {
#             'mean': round(merged_df['Water Level (mm)'].mean(), 2),
#             'std': round(merged_df['Water Level (mm)'].std(), 2),
#             'min': round(merged_df['Water Level (mm)'].min(), 2),
#             'max': round(merged_df['Water Level (mm)'].max(), 2),
#             'total_readings': len(merged_df),
#             'unique_farms': merged_df['Farm ID'].nunique()
#         }
#         context['farm_summary_stats'] = stats
        
#         # Reload the page with updated context
#         return render(request, 'water_meters.html', context)

#     # ----------------------------------
#     # 8️⃣ Handle Comparative Analysis Downloads
#     # ----------------------------------
#     if request.method == 'GET' and any(k in request.GET for k in [
#     'download_village_analysis', 'download_rc_analysis', 'download_awd_analysis', 'download_dsr_tpr_analysis', 'download_custom_group_analysis'
#     ]):
#        section_map = {
#         'download_village_analysis': 'village',
#         'download_rc_analysis': 'rc',
#         'download_awd_analysis': 'awd',
#         'download_dsr_tpr_analysis': 'dsr_tpr',
#         'download_custom_group_analysis': 'custom'
#     }
#        for key, section in section_map.items():
#         if key in request.GET:
#             merged_df = pd.read_json(session['merged_df'], orient='split')
#             kharif_df = pd.read_json(session['kharif_df'], orient='split')
#             custom_groups = session.get('custom_groups', {})
#             custom_groups_df = {k: pd.read_json(StringIO(v), orient='split') for k, v in custom_groups.items()}
#             summary_df = get_comparative_analysis(section, merged_df, kharif_df, custom_groups_df)
#             if summary_df is not None:
#                 summary_csv = summary_df.to_csv(index=False)
#                 return HttpResponse(summary_csv, content_type='text/csv', headers={
#                     'Content-Disposition': f'attachment; filename={section}_analysis.csv'
#                 })
#             else:
#                 return HttpResponse(b"No data available.", status=400)
#     # ----------------------------------
#     # 🖼️ Final Page Render
#     # ----------------------------------
#     return render(request, 'water_meters.html', context)


# # Helper for pretty download names
# def pretty_download_name(filename):
#     mapping = {
#         "01_complete_merged_data.csv": "Complete Merged Dataset",
#         "02_raw_kharif_data.csv": "Raw Kharif Data",
#         "20_village_summary.csv": "Village Summary",
#         "21_farm_summary.csv": "Farm Summary",
#         "30_remote_controller_summary.csv": "Remote Controller Summary",
#         "31_awd_study_summary.csv": "AWD Study Summary",
#         "32_dsr_vs_tpr_summary.csv": "DSR vs TPR Summary",
#     }
#     # For study group files
#     if filename.endswith(".csv") and filename[3:-4]:
#         name = filename[3:-4].replace("_", " ").title()
#         if "Remote Controllers Treatment" in name:
#             return "Remote Controllers Treatment"
#         if "Remote Controllers Control" in name:
#             return "Remote Controllers Control"
#         if "Awd Group A Treatment" in name:
#             return "AWD Group A Treatment"
#         if "Awd Group B Training" in name:
#             return "AWD Group B Training"
#         if "Awd Group C Control" in name:
#             return "AWD Group C Control"
#         if "Dsr Farms" in name:
#             return "DSR Farms"
#         if "Tpr Farms" in name:
#             return "TPR Farms"
#         return name
#     return mapping.get(filename, filename)



# @require_auth
# def farmer_survey(request):
#     return render(request, 'farmer_survey.html')


# @require_auth
# def evapotranspiration(request):
#     return render(request, 'evapotranspiration.html')


# @require_auth
# def mapping(request):
#     return render(request, 'mapping.html')


# # digivi/views.py

# from django.shortcuts import render
# import pandas as pd

# from .utils import kharif2025_farms, get_2025plots, get_meters_by_village



# # Replace the existing meter_reading_25_view function with this updated version:
# @require_auth
# def meter_reading_25_view(request):
#     # ANSI color codes for terminal output
#     class Colors:
#         GREEN = '\033[92m'
#         RED = '\033[91m'
#         YELLOW = '\033[93m'
#         BLUE = '\033[94m'
#         CYAN = '\033[96m'
#         WHITE = '\033[97m'
#         BOLD = '\033[1m'
#         END = '\033[0m'

#     def print_status(message, status="info"):
#         """Print colored status messages with icons"""
#         if status == "success":
#             print(f"{Colors.GREEN}✓ {message}{Colors.END}")
#         elif status == "error":
#             print(f"{Colors.RED}✗ {message}{Colors.END}")
#         elif status == "warning":
#             print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
#         elif status == "info":
#             print(f"{Colors.BLUE}ℹ {message}{Colors.END}")
#         elif status == "process":
#             print(f"{Colors.CYAN}⚡ {message}{Colors.END}")

#     # Your Django view code with visual feedback
#     error = None
#     farm_ids = []
#     village_names = []
#     selected = request.POST.get('selected_farm', '')
#     selected_village = request.POST.get('selected_village', '')
#     results = []

#     print_status("Initializing view variables", "info")

#     # Date filter variables
#     use_date_filter = request.POST.get('use_date_filter') == 'on' or request.session.get('use_date_filter', False)
#     filter_start_date = request.POST.get('filter_start_date') or request.session.get('filter_start_date')
#     filter_end_date = request.POST.get('filter_end_date') or request.session.get('filter_end_date')

#     if use_date_filter:
#         print_status(f"Date filter enabled: {filter_start_date} to {filter_end_date}", "info")
#     else:
#         print_status("Date filter disabled", "info")

#     # 1) Upload & cache raw readings + Load master from Google Sheets
#     if request.method == 'POST' and 'raw_file' in request.FILES:
#         try:
#             print_status("Processing raw file upload...", "process")
            
#             # Load raw file
#             print_status("Reading Excel file...", "process")
#             raw_df = pd.read_excel(request.FILES['raw_file'])
#             print_status(f"Raw file loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns", "success")
            
#             # Clean raw data to handle empty entries
#             print_status("Cleaning raw data...", "process")
#             raw_df_cleaned = raw_df.replace(r'^\s*$', pd.NA, regex=True)  # Empty or whitespace-only
#             raw_df_cleaned = raw_df_cleaned.replace('', pd.NA)  # Explicit empty strings
#             raw_df_cleaned = raw_df_cleaned.replace('nan', pd.NA)  # String 'nan'
#             raw_df_cleaned = raw_df_cleaned.replace('NaN', pd.NA)  # String 'NaN'
#             print_status("Empty values cleaned", "success")
            
#             # Convert columns to appropriate data types where possible
#             print_status("Converting data types...", "process")
#             numeric_cols = 0
#             for col in raw_df_cleaned.columns:
#                 original_type = raw_df_cleaned[col].dtype
#                 raw_df_cleaned[col] = pd.to_numeric(raw_df_cleaned[col], errors='ignore')
#                 if raw_df_cleaned[col].dtype != original_type and pd.api.types.is_numeric_dtype(raw_df_cleaned[col]):
#                     numeric_cols += 1
            
#             if numeric_cols > 0:
#                 print_status(f"Converted {numeric_cols} columns to numeric types", "success")
#             else:
#                 print_status("No columns converted to numeric", "info")
            
#             print_status("Caching raw data in session...", "process")
#             request.session['raw25'] = raw_df_cleaned.to_json(date_format='iso')
#             print_status("Raw data cached successfully", "success")
        
#             # Load master data from Google Sheets
#             print_status("Loading master data from Google Sheets...", "process")
#             master25 = load_master_from_google_sheets()
            
#             if master25:
#                 print_status(f"Master data loaded from Google Sheets: {len(master25)} sheets", "success")
                
#                 # Clean and process master data to handle empty entries
#                 print_status("Processing master data sheets...", "process")
#                 cleaned_master = {}
                
#                 for name, df in master25.items():
#                     print_status(f"Cleaning sheet: {name}", "process")
                    
#                     # Replace empty strings, whitespace-only strings, and 'nan' strings with actual NaN
#                     df_cleaned = df.replace(r'^\s*$', pd.NA, regex=True)  # Empty or whitespace-only
#                     df_cleaned = df_cleaned.replace('', pd.NA)  # Explicit empty strings
#                     df_cleaned = df_cleaned.replace('nan', pd.NA)  # String 'nan'
#                     df_cleaned = df_cleaned.replace('NaN', pd.NA)  # String 'NaN'
                    
#                     # Convert columns to appropriate data types where possible
#                     # This helps ensure numeric columns don't have string representations of numbers
#                     numeric_conversions = 0
#                     for col in df_cleaned.columns:
#                         original_type = df_cleaned[col].dtype
#                         # Try to convert to numeric, keeping NaN for non-numeric values
#                         df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
#                         if df_cleaned[col].dtype != original_type and pd.api.types.is_numeric_dtype(df_cleaned[col]):
#                             numeric_conversions += 1
                    
#                     cleaned_master[name] = df_cleaned
#                     print_status(f"Sheet '{name}' processed: {len(df_cleaned)} rows, {numeric_conversions} numeric columns", "success")
                
#                 print_status("Caching master data in session...", "process")
#                 request.session['master25'] = {
#                     name: df.to_json(date_format='iso') for name, df in cleaned_master.items()
#                 }
#                 print_status("Master data cached successfully", "success")
#             else:
#                 error = "Failed to load master data from Google Sheets"
#                 print_status(error, "error")
        
#             # Clear date filter when new file is uploaded
#             print_status("Clearing previous date filters...", "process")
#             request.session['use_date_filter'] = False
#             request.session['filter_start_date'] = None
#             request.session['filter_end_date'] = None
#             print_status("Date filters cleared", "success")
            
#             print_status("File upload and processing completed successfully", "success")
            
#         except Exception as e:
#             error = f"Error processing files: {e}"
#             print_status(error, "error")

#     # 2) Manual master workbook upload (fallback option)
#     if request.method == 'POST' and 'meters_file' in request.FILES:
#         try:
#             print_status("Processing manual master workbook upload...", "process")
            
#             print_status("Reading Excel workbook with all sheets...", "process")
#             master = pd.read_excel(request.FILES['meters_file'], sheet_name=None)
#             print_status(f"Master workbook loaded: {len(master)} sheets", "success")
            
#             # Apply same cleaning logic to manual uploads
#             print_status("Cleaning master workbook data...", "process")
#             cleaned_master = {}
            
#             for name, df in master.items():
#                 print_status(f"Processing sheet: {name}", "process")
                
#                 # Replace empty strings, whitespace-only strings, and 'nan' strings with actual NaN
#                 df_cleaned = df.replace(r'^\s*$', pd.NA, regex=True)  # Empty or whitespace-only
#                 df_cleaned = df_cleaned.replace('', pd.NA)  # Explicit empty strings
#                 df_cleaned = df_cleaned.replace('nan', pd.NA)  # String 'nan'
#                 df_cleaned = df_cleaned.replace('NaN', pd.NA)  # String 'NaN'
                
#                 # Convert columns to appropriate data types where possible
#                 numeric_conversions = 0
#                 for col in df_cleaned.columns:
#                     original_type = df_cleaned[col].dtype
#                     df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='ignore')
#                     if df_cleaned[col].dtype != original_type and pd.api.types.is_numeric_dtype(df_cleaned[col]):
#                         numeric_conversions += 1
                
#                 cleaned_master[name] = df_cleaned
#                 print_status(f"Sheet '{name}' cleaned: {len(df_cleaned)} rows, {numeric_conversions} numeric columns", "success")
            
#             print_status("Caching cleaned master data in session...", "process")
#             request.session['master25'] = {
#                 name: df.to_json(date_format='iso') for name, df in cleaned_master.items()
#             }
#             print_status("Manual master workbook processed and cached successfully", "success")
            
#         except Exception as e:
#             error = f"Error reading master file: {e}"
#             print_status(error, "error")

#     # 3) Handle date filter update
#     if request.method == 'POST' and 'update_date_filter' in request.POST:
#         use_date_filter = request.POST.get('use_date_filter') == 'on'
#         request.session['use_date_filter'] = use_date_filter
        
#         if use_date_filter:
#             filter_start_date = request.POST.get('filter_start_date')
#             filter_end_date = request.POST.get('filter_end_date')
#             request.session['filter_start_date'] = filter_start_date
#             request.session['filter_end_date'] = filter_end_date
#         else:
#             request.session['filter_start_date'] = None
#             request.session['filter_end_date'] = None
        
#         # Redirect to same page to refresh with new filter
#         return redirect('meter_reading_25')

#     # 4) Retrieve from session
#     raw_df = pd.read_json(request.session['raw25']) if 'raw25' in request.session else None
#     master25 = {
#         name: pd.read_json(json_str)
#         for name, json_str in request.session.get('master25', {}).items()
#     } if 'master25' in request.session else None
    
#     # Get date range for display
#     date_range_info = None
#     if raw_df is not None:
#         raw_df['Date'] = pd.to_datetime(raw_df['Date'], dayfirst=False)
#         min_date = raw_df['Date'].min().strftime('%Y-%m-%d')
#         max_date = raw_df['Date'].max().strftime('%Y-%m-%d')
#         date_range_info = {
#             'min': min_date,
#             'max': max_date,
#             'current_min': min_date,
#             'current_max': max_date
#         }
        
#         # Apply date filter if enabled
#         if use_date_filter and filter_start_date and filter_end_date:
#             date_range_info['current_min'] = filter_start_date
#             date_range_info['current_max'] = filter_end_date

#     # 5) Build farm dropdown and village dropdown
#     if master25:
#         farm_dict = kharif2025_farms(master25)
#         farm_ids = list(kharif2025_farms(master25).keys())
        
#         # Add village extraction
#         if raw_df is not None:
#             # Column D is index 3 (0-based)
#             village_names = sorted(raw_df.iloc[:, 3].dropna().unique().tolist())
        
#         download_request = request.POST.get("download_table")
#         if download_request and raw_df is not None:
#             col_to_get = "m³ per Acre per Avg Day" if download_request == "avg" else "m³ per Acre"
#             if use_date_filter and filter_start_date and filter_end_date:
#                 filter_start = pd.to_datetime(filter_start_date)
#                 filter_end = pd.to_datetime(filter_end_date)
#                 combined_df = get_tables(raw_df, master25, farm_dict, col_to_get, start_date_enter=filter_start, end_date_enter=filter_end)
#             else:
#                 combined_df = get_tables(raw_df, master25, farm_dict, col_to_get)

#             # Convert DataFrame to Excel file in memory
#             with BytesIO() as buffer:
#                 with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
#                     combined_df.to_excel(writer, index=False, sheet_name='Combined Table')
#                 buffer.seek(0)
#                 filename = f"table_{download_request}_kharif2025.xlsx"
#                 response = HttpResponse(buffer.getvalue(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
#                 response['Content-Disposition'] = f'attachment; filename="{filename}"'
#                 return response

#     # Handle Word report download
#     if request.method == 'POST' and request.POST.get('download_report') and raw_df is not None and master25:
#         # Reconstruct results based on what was selected - GENERATE MATPLOTLIB FOR REPORTS
#         filter_type = None
#         filter_value = None
#         report_results = []
        
#         if request.POST.get('report_filter_type') == 'farm':
#             selected = request.POST.get('report_filter_value')
#             filter_type = "Farm Filter"
#             filter_value = selected
#             mapping = kharif2025_farms(master25)
#             meters = mapping.get(selected, [])
#             if use_date_filter and filter_start_date and filter_end_date:
#                 filter_start = pd.to_datetime(filter_start_date)
#                 filter_end = pd.to_datetime(filter_end_date)
#                 encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
#             else:
#                 encoded_imgs = get_2025plots(raw_df, master25, selected, meters)
            
#             for idx, meter in enumerate(meters):
#                 block = {
#                     'meter': meter,
#                     'plots': encoded_imgs[4*idx : 4*idx + 4]
#                 }
#                 report_results.append(block)
                
#         elif request.POST.get('report_filter_type') == 'village':
#             selected_village = request.POST.get('report_filter_value')
#             filter_type = "Village Filter"
#             filter_value = selected_village
#             from .utils import get_meters_by_village
            
#             village_meters = get_meters_by_village(raw_df, selected_village)
#             farm_dict = kharif2025_farms(master25)
#             meter_to_farm = {}
#             for farm_id, meter_list in farm_dict.items():
#                 for meter in meter_list:
#                     meter_to_farm[meter] = farm_id
            
#             for meter in village_meters:
#                 if meter in meter_to_farm:
#                     farm_id = meter_to_farm[meter]

#                     if use_date_filter and filter_start_date and filter_end_date:
#                         filter_start = pd.to_datetime(filter_start_date)
#                         filter_end = pd.to_datetime(filter_end_date)
#                         meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
#                     else:
#                         meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter])
#                     for idx in range(0, len(meter_imgs), 4):
#                         block = {
#                             'meter': meter,
#                             'farm': farm_id,
#                             'plots': meter_imgs[idx:idx+4]
#                         }
#                         report_results.append(block)
        
#         # Generate Word report
#         from .utils import generate_word_report
#         docx_buffer = generate_word_report(report_results, filter_type, filter_value, raw_df, master25)
        
#         # Return Word document
#         response = HttpResponse(
#             docx_buffer.getvalue(),
#             content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
#         )
#         response['Content-Disposition'] = f'attachment; filename="water_analysis_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.docx"'
#         return response

#     # 6) When a farm is selected, generate BOTH matplotlib (for reports) and plotly (for display)
#     if selected and raw_df is not None and master25:
#         mapping = kharif2025_farms(master25)
#         meters = mapping.get(selected, [])
        
#         # Generate PLOTLY plots for portal display
#         if use_date_filter and filter_start_date and filter_end_date:
#             filter_start = pd.to_datetime(filter_start_date)
#             filter_end = pd.to_datetime(filter_end_date)
#             plotly_htmls = get_2025plots_plotly(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
#             # Also generate matplotlib for potential reports
#             encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
#         else:
#             plotly_htmls = get_2025plots_plotly(raw_df, master25, selected, meters)
#             # Also generate matplotlib for potential reports
#             encoded_imgs = get_2025plots(raw_df, master25, selected, meters)

#         # Group plots per meter - 4 plotly plots per meter
#         for idx, meter in enumerate(meters):
#             block = {
#                 'meter': meter,
#                 'plotly_plots': plotly_htmls[4*idx : 4*idx + 4],  # Plotly for display
#                 'plots': encoded_imgs[4*idx : 4*idx + 4],         # Matplotlib for reports
#                 'is_combined': False
#             }
#             results.append(block)
        
#         # Generate combined plots if multiple meters
#         if len(meters) > 1:
#             if use_date_filter and filter_start_date and filter_end_date:
#                 filter_start = pd.to_datetime(filter_start_date)
#                 filter_end = pd.to_datetime(filter_end_date)
#                 combined_plotly = get_2025plots_combined_plotly(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
#                 combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
#             else:
#                 combined_plotly = get_2025plots_combined_plotly(raw_df, master25, selected, meters)
#                 combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters)
            
#             if combined_plotly:
#                 combined_block = {
#                     'meter': ' + '.join(meters),
#                     'plotly_plots': combined_plotly,  # Plotly for display
#                     'plots': combined_imgs,           # Matplotlib for reports
#                     'is_combined': True,
#                     'farm': selected
#                 }
#                 results.append(combined_block)
    
#     # 7) When a village is selected, generate graphs for all meters in that village
#     elif selected_village and raw_df is not None and master25:
#         from .utils import get_meters_by_village
        
#         # Get all meters for this village
#         village_meters = get_meters_by_village(raw_df, selected_village)
        
#         # Create reverse mapping of meter to farm
#         farm_dict = kharif2025_farms(master25)
#         meter_to_farm = {}
#         for farm_id, meter_list in farm_dict.items():
#             for meter in meter_list:
#                 meter_to_farm[meter] = farm_id
        
#         # Group meters by farm for combined analysis
#         farm_meters_map = {}
#         for meter in village_meters:
#             if meter in meter_to_farm:
#                 farm_id = meter_to_farm[meter]
#                 if farm_id not in farm_meters_map:
#                     farm_meters_map[farm_id] = []
#                 farm_meters_map[farm_id].append(meter)
        
#         # Generate plots for each farm in the village
#         for farm_id, farm_meters in farm_meters_map.items():
#             # Individual meter plots
#             for meter in farm_meters:
#                 if use_date_filter and filter_start_date and filter_end_date:
#                     filter_start = pd.to_datetime(filter_start_date)
#                     filter_end = pd.to_datetime(filter_end_date)
#                     meter_plotly = get_2025plots_plotly(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
#                     meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
#                 else:
#                     meter_plotly = get_2025plots_plotly(raw_df, master25, farm_id, [meter])
#                     meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter])
                
#                 for idx in range(0, len(meter_plotly), 4):
#                     block = {
#                         'meter': meter,
#                         'farm': farm_id,
#                         'plotly_plots': meter_plotly[idx:idx+4],  # Plotly for display
#                         'plots': meter_imgs[idx:idx+4],           # Matplotlib for reports
#                         'is_combined': False
#                     }
#                     results.append(block)
            
#             # Combined plots if multiple meters for this farm
#             if len(farm_meters) > 1:
#                 if use_date_filter and filter_start_date and filter_end_date:
#                     filter_start = pd.to_datetime(filter_start_date)
#                     filter_end = pd.to_datetime(filter_end_date)
#                     combined_plotly = get_2025plots_combined_plotly(raw_df, master25, farm_id, farm_meters, start_date_enter=filter_start, end_date_enter=filter_end)
#                     combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters, start_date_enter=filter_start, end_date_enter=filter_end)
#                 else:
#                     combined_plotly = get_2025plots_combined_plotly(raw_df, master25, farm_id, farm_meters)
#                     combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters)
                
#                 if combined_plotly:
#                     combined_block = {
#                         'meter': ' + '.join(farm_meters),
#                         'farm': farm_id,
#                         'plotly_plots': combined_plotly,  # Plotly for display
#                         'plots': combined_imgs,           # Matplotlib for reports
#                         'is_combined': True
#                     }
#                     results.append(combined_block)

#     return render(request, 'meter_reading_25.html', {
#         'error': error,
#         'farm_ids': farm_ids,
#         'village_names': village_names,
#         'selected': selected,
#         'selected_village': selected_village,
#         'results': results,
#         'date_range_info': date_range_info,
#         'use_date_filter': use_date_filter,
#         'filter_start_date': filter_start_date,
#         'filter_end_date': filter_end_date,
#     })
# @require_auth
# def grouping_25(request):
#     if request.method == 'POST':
#         selected_label = request.POST.get('group_type')
#         selected_checkboxes = request.POST.getlist('group_category')

#         raw_df = pd.read_json(request.session['raw25']) if 'raw25' in request.session else None
#         master25 = {
#             name: pd.read_json(json_str)
#             for name, json_str in request.session.get('master25', {}).items()
#         } if 'master25' in request.session else None
        
#         # Apply date filter if enabled
#         if raw_df is not None and request.session.get('use_date_filter', False):
#             start_date = request.session.get('filter_start_date')
#             end_date = request.session.get('filter_end_date')

       

#         # Handle report download
#         if request.POST.get('download_group_report') and 'group_plot' in request.session:
#             from .utils import generate_group_analysis_report
            
#             # Retrieve stored data from session
#             stored_data = request.session.get('group_analysis_data', {})
            
#             docx_buffer = generate_group_analysis_report(
#                 stored_data.get('group_type', ''),
#                 stored_data.get('selected_groups', []),
#                 request.session.get('group_plot', ''),
#                 request.session.get('group_plot2', ''),
#                 stored_data.get('group_farms', {})
#             )
            
#             response = HttpResponse(
#                 docx_buffer.getvalue(),
#                 content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
#             )
#             response['Content-Disposition'] = f'attachment; filename="group_analysis_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.docx"'
#             return response

#         if selected_label and selected_checkboxes and raw_df is not None and master25:
#             kharif_df = master25['Farm details']

#             group_farms_dict = {}

#             # Map checkbox labels to column names in master file
#             group_column_map = {
#                 "Remote": {
#                     "Group-A Complied": "Kharif 25 - Remote Controllers Study - Group A - Treatment - complied (Y/N)",
#                     "Group-A Non-Complied": "Kharif 25 - Remote Controllers Study - Group A - Treatment - NON-complied (Y/N)",
#                     "Group-B Complied": "Kharif 25 - Remote Controllers Study - Group B - Control - complied (Y/N)",
#                     "Group-B Non-Complied": "Kharif 25 - Remote Controllers Study - Group B - Control - NON-complied (Y/N)",
#                 },
#                 "AWD": {
#                     "Group-A Complied": "Kharif 25 - AWD Study - Group A - Treatment - complied (Y/N)",
#                     "Group-A Non-Complied": "Kharif 25 - AWD Study - Group A - Treatment - Non-complied (Y/N)",
#                     "Group-B Complied": "Kharif 25 - AWD Study - Group B - Complied (Y/N)",
#                     "Group-B Non-Complied": "Kharif 25 - AWD Study - Group B - Non-complied (Y/N)",
#                     "Group-C Complied": "Kharif 25 - AWD Study - Group C - Complied (Y/N)",
#                     "Group-C Non-Complied": "Kharif 25 - AWD Study - Group C - non-complied (Y/N)",
#                 },
#                 "TPR/DSR": {
#                     "TPR": "Kharif 25 - TPR Group Study (Y/N)",
#                     "DSR": "Kharif 25 - DSR farm Study (Y/N)",
#                 }
#             }

#             simplified_groups = {}
#             for group in selected_checkboxes:
#                 base = group.split()[0] 
#                 label = selected_label + " " + base  
#                 if label not in simplified_groups:
#                     simplified_groups[label] = []
#                 simplified_groups[label].append(group)

#             # Build group-wise farm ID lists
#             for label, group_list in simplified_groups.items():
#                 cols = [group_column_map[selected_label][g] for g in group_list]
#                 condition = (kharif_df[cols[0]].fillna(0) == 1)
#                 for c in cols[1:]:
#                     condition |= (kharif_df[c].fillna(0) == 1)
#                 farm_ids = kharif_df.loc[condition, "Kharif 25 Farm ID"].tolist()
#                 group_farms_dict[label] = farm_ids

#             # Calculate and merge averages
#             group_dfs = []
#             group_dfs2 = []
#             for label, farms in group_farms_dict.items():
#                 if raw_df is not None and request.session.get('use_date_filter', False):
#                     filter_start = pd.to_datetime(start_date)
#                     filter_end = pd.to_datetime(end_date)
#                     df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, 'm³ per Acre per Avg Day' ,start_date_enter=filter_start, end_date_enter=filter_end)
#                     df2 = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, "Delta m³" ,start_date_enter=filter_start, end_date_enter=filter_end)
#                 else:
#                     df = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, 'm³ per Acre per Avg Day')
#                     df2 = calculate_avg_m3_per_acre(selected_label, label, farms, raw_df, master25, "Delta m³")
#                 group_dfs.append(df)
#                 group_dfs2.append(df2)

#             # Merge all into one plot
#             if group_dfs:
#                 final_df = group_dfs[0]
#                 for df in group_dfs[1:]:
#                     final_df = pd.merge(final_df, df, on="Day", how="outer")
#                 group_plot = generate_group_analysis_plot(final_df, "Daily Average m3/acre")
                
#                 # Store in session for download
#                 request.session['group_plot'] = group_plot
#                 request.session['group_analysis_data'] = {
#                     'group_type': selected_label,
#                     'selected_groups': selected_checkboxes,
#                     'group_farms': group_farms_dict
#                 }
            
#             group_plot2 = None
#             if group_dfs2:
#                 final_df2 = group_dfs2[0]
#                 for df in group_dfs2[1:]:
#                     final_df2 = pd.merge(final_df2, df, on="Day", how="outer")
#                 group_plot2 = generate_group_analysis_plot(final_df2, "Delta m3/acre")
                
#                 # Store in session for download
#                 request.session['group_plot2'] = group_plot2
               
#             return render(request, 'grouping.html', {
#                 'group_plot': group_plot,
#                 'group_plot2': group_plot2,
#                 'output': True,
#                 'group_type': selected_label,
#                 'selected_groups': selected_checkboxes,
#             })
    
#     return render(request, 'grouping.html')

# @csrf_exempt
# def api_token(request):
#     if request.method == 'POST':
#         import json
#         data = json.loads(request.body.decode())
#         username = data.get('username')
#         password = data.get('password')
#         if username == API_USERNAME and password == API_PASSWORD:
#             payload = {
#                 'username': username,
#                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
#                 'iat': datetime.datetime.utcnow(),
#             }
#             token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
#             return JsonResponse({'access': token})
#         else:
#             return JsonResponse({'detail': 'Invalid credentials'}, status=401)
#     return JsonResponse({'detail': 'Method not allowed'}, status=405)

# #VIEWS FOR KNOWLEDGE BASE SECTION TEMPLATE FILES

# def agriculture_view(request):
#     return render(request, 'agriculture.html')

# def crop_view(request):
#     return render(request, 'crop_residue.html')

# def dsr_view(request):
#     return render(request, 'dsr.html')

# def farmers_view(request):
#     return render(request, 'farmers.html')

# def stages_view(request):
#     return render(request, 'stages.html')

# def tpr_view(request):
#     return render(request, 'tpr.html')

# def tubewell_view(request):
#     return render(request, 'tubewell.html')





# digivi/views.py

from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
import pandas as pd
from io import BytesIO, StringIO
from django.http import HttpResponse, JsonResponse
import datetime  # Add this import
import io
import zipfile
import re
import jwt
# In views.py, update the imports section:
from .utils import (
    kharif2024_farms, get_2024plots,
    kharif2025_farms, get_2025plots, get_2025plots_combined,
    get_2025plots_plotly, get_2025plots_combined_plotly, 
    encode_plot_to_base64, get_tables,
    calculate_avg_m3_per_acre, generate_group_analysis_plot,
    load_and_validate_data,
    clean_and_process_data,
    create_merged_dataset,
    apply_comprehensive_filters,
    render_comparative_analysis,
    get_farm_analysis_data,
    create_zip_package,
    prepare_comprehensive_downloads,
    get_per_farm_downloads,
    get_village_level_analysis,
    load_master_from_google_sheets,
    print_status,
    generate_delta_vs_days_groupwise_plots
)
from django.urls import reverse
from functools import wraps
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
import jwt
from django.views.decorators.csrf import csrf_exempt
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
        # Handle both JSON and form data
        if request.content_type == 'application/json':
            try:
                import json
                data = json.loads(request.body.decode())
                username = data.get('username')
                password = data.get('password')
            except (json.JSONDecodeError, UnicodeDecodeError):
                username = None
                password = None
        else:
            username = request.POST.get('username')
            password = request.POST.get('password')
        
        # Debug logging
        print(f"[DEBUG] Login attempt - Username: '{username}', Password: '{password}'")
        print(f"[DEBUG] Expected - Username: '{LOGIN_USERNAME}', Password: '{LOGIN_PASSWORD}'")
        print(f"[DEBUG] Username match: {username == LOGIN_USERNAME}")
        print(f"[DEBUG] Password match: {password == LOGIN_PASSWORD}")
        
        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            payload = {
                'username': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
                'iat': datetime.datetime.utcnow(),
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
            print(f"[DEBUG] Login successful, redirecting to index")
            
            # Handle JSON response
            if request.content_type == 'application/json':
                from django.http import JsonResponse
                response = JsonResponse({'success': True, 'message': 'Login successful'})
                response.set_cookie(JWT_COOKIE_NAME, token, httponly=True, samesite='Lax')
                return response
            else:
                response = redirect('index')
                response.set_cookie(JWT_COOKIE_NAME, token, httponly=True, samesite='Lax')
                return response
        else:
            error = 'Invalid credentials.'
            print(f"[DEBUG] Login failed - Invalid credentials")
            
            # Handle JSON error response
            if request.content_type == 'application/json':
                from django.http import JsonResponse
                return JsonResponse({'success': False, 'message': 'Invalid credentials'})
    
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

def get_comparative_analysis(section, merged_df, kharif_df, custom_groups_df):
    comparative = render_comparative_analysis(merged_df, kharif_df, custom_groups_df)
    if section == 'village':
        return comparative['village']['summary_df']
    elif section == 'rc':
        return comparative['rc']['summary_df']
    elif section == 'awd':
        return comparative['awd']['summary_df']
    elif section == 'dsr_tpr':
        return comparative['dsr_tpr']['summary_df']
    elif section == 'custom':
        return comparative['custom']['summary_df']
    else:
        return None
    

def sanitize_filename(filename):
    """Sanitize filename for safe file operations"""
    return re.sub(r'[^\w\-_\.]', '_', str(filename))

def load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session):
    """Load all necessary context to prevent page breaking when errors occur"""
    def categorize_files(downloadables):
        complete_dataset = [f for f in downloadables if f.startswith('01_') or f.startswith('02_')]
        study_group = [f for f in downloadables if any(k in f.lower() for k in ['remote', 'awd', 'dsr', 'tpr'])]
        analysis_reports = [f for f in downloadables if 'village_summary' in f or 'farm_summary' in f]
        return complete_dataset, study_group, analysis_reports
 

from io import StringIO
from django.shortcuts import render
from django.http import HttpResponse
from .utils import (
    load_and_validate_data,
    clean_and_process_data,
    create_merged_dataset,
    apply_comprehensive_filters,
    render_comparative_analysis,
    get_farm_analysis_data,
    create_zip_package,
    prepare_comprehensive_downloads,
    get_per_farm_downloads,
    get_village_level_analysis
)
import pandas as pd
import json
import io
import zipfile
from io import StringIO
import re
def water_dashboard_view(request):
    context = {}
    session = request.session

    def categorize_files(downloadables):
        complete_dataset = [f for f in downloadables if f.startswith('01_') or f.startswith('02_')]
        study_group = [f for f in downloadables if any(k in f.lower() for k in ['remote', 'awd', 'dsr', 'tpr'])]
        analysis_reports = [f for f in downloadables if 'village_summary' in f or 'farm_summary' in f]
        return complete_dataset, study_group, analysis_reports

    # Handle date range filtering
    use_date_filter = request.POST.get('use_date_filter') == 'on' or session.get('use_date_filter', False)
    start_date = request.POST.get('start_date') or session.get('start_date')
    end_date = request.POST.get('end_date') or session.get('end_date')
    
    # Store date filter settings in session
    if 'update_date_filter' in request.POST:
        use_date_filter = request.POST.get('use_date_filter') == 'on'
        session['use_date_filter'] = use_date_filter
        if use_date_filter:
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')
            session['start_date'] = start_date
            session['end_date'] = end_date
        else:
            session['start_date'] = None
            session['end_date'] = None
            start_date = None
            end_date = None
        session.modified = True
    
    # Pass date filter settings to context
    context['use_date_filter'] = use_date_filter
    context['start_date'] = start_date
    context['end_date'] = end_date

    def load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session, comparative_results=None):
        """Helper function to load complete context data"""
        # Basic data setup
        context['merged_loaded'] = True
        context['available_farms'] = sorted(merged_df["Farm ID"].unique())
        context['available_villages'] = sorted(merged_df["Village"].dropna().unique()) if 'Village' in merged_df.columns else []
        context['selected_farm'] = context.get('selected_farm') or (context['available_farms'][0] if context['available_farms'] else None)
        
        # Load custom groups
        custom_groups = session.get('custom_groups', {})
        custom_groups_df = {}
        for k, v in custom_groups.items():
            try:
                df = pd.read_json(StringIO(v), orient='split')
                if not df.empty:
                    required = {"Farm ID", "Village", "Water Level (mm)", "Date", "Days from TPR"}
                    if required.issubset(df.columns):
                        custom_groups_df[k] = df
            except Exception as e:
                print(f"[ERROR] Failed to parse custom group '{k}':", e)
        
        context['custom_groups'] = list(custom_groups_df.keys())
        
        # Generate comparative analysis if not provided
        if comparative_results is None:
            selected_villages_comparison = session.get('selected_villages_comparison', None)
            selected_rc_groups = session.get('selected_rc_groups', None)
            selected_awd_groups = session.get('selected_awd_groups', None)
            comparative_results = render_comparative_analysis(merged_df, kharif_df, custom_groups_df, selected_villages_comparison, selected_rc_groups, selected_awd_groups)
        
        # Handle village analysis with error checking
        village_plot_html = ""
        village_summary_html = ""
        
        if "error" not in comparative_results['village']:
            village_plot_html = comparative_results['village']['plot'].to_html(full_html=False)
            village_summary_html = comparative_results['village']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
        else:
            village_plot_html = f"<div class='alert alert-warning'>{comparative_results['village']['error']}</div>"
            village_summary_html = "<div class='text-muted'>No village data available.</div>"
        
        context['comparative'] = {
            'village': {
                'plot': village_plot_html,
                'summary_df': village_summary_html
            },
            'rc': {
                'plot': comparative_results['rc']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['rc']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            },
            'awd': {
                'plot': comparative_results['awd']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['awd']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            },
            'dsr_tpr': {
                'plot': comparative_results['dsr_tpr']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['dsr_tpr']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            },
            'compliance': {
                'df': comparative_results['compliance']['df'].to_html(index=False, classes="table table-striped"),
                'summary': comparative_results['compliance']['summary']
            },
            'custom': {
                'summary_df': (
                    comparative_results['custom']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
                    if comparative_results['custom'].get('summary_df') is not None and not comparative_results['custom']['summary_df'].empty
                    else "<div class='text-muted'>No valid custom groups available.</div>"
                ),
                'chart_html': comparative_results['custom'].get('chart_html', None)
            }
        }
        
        # Download categories
        downloads = prepare_comprehensive_downloads(merged_df, kharif_df, farm_daily_avg, weekly_avg)
        ds1, ds2, ds3 = categorize_files(downloads)
        context['complete_dataset_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds1]
        context['study_group_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds2]
        context['analysis_report_files'] = [{'filename': f, 'display_name': pretty_download_name(f)} for f in ds3]
        
        # Farm summary statistics
        stats = {
            'mean': round(merged_df['Water Level (mm)'].mean(), 2),
            'std': round(merged_df['Water Level (mm)'].std(), 2),
            'min': round(merged_df['Water Level (mm)'].min(), 2),
            'max': round(merged_df['Water Level (mm)'].max(), 2),
            'total_readings': len(merged_df),
            'unique_farms': merged_df['Farm ID'].nunique()
        }
        context['farm_summary_stats'] = stats
        
        # Per-farm downloads
        if context['selected_farm']:
            context['per_farm_downloads'] = [
                {'filename': f"farm_{sanitize_filename(context['selected_farm'])}_{suffix}.csv", 'display_name': label}
                for suffix, label in zip(['detailed', 'summary', 'daily_avg'], ["Detailed Data", "Summary Row", "Daily Averages"])
            ]
        else:
            context['per_farm_downloads'] = []
        
        # Farm analysis if a farm is selected
        if context['selected_farm']:
            analysis = get_farm_analysis_data(context['selected_farm'], merged_df, farm_daily_avg, weekly_avg)
            context['farm_analysis'] = {
                "plot1": analysis["plot1"],
                "plot2": analysis["plot2"],
                "summary": analysis["summary"],
                "pivot_table_html": analysis["pivot_table"]
            }
        
        # Add selected items for comparison to context
        context['selected_villages_comparison'] = session.get('selected_villages_comparison', [])
        context['selected_rc_groups'] = session.get('selected_rc_groups', [])
        context['selected_awd_groups'] = session.get('selected_awd_groups', [])
        
        # Add available RC and AWD groups for the selection forms
        context['available_rc_groups'] = [
            "Treatment Group (A) - Complied",
            "Treatment Group (A) - Non-Complied", 
            "Control Group (B) - Complied",
            "Control Group (B) - Non-Complied"
        ]
        context['available_awd_groups'] = [
            "Group A (Treatment) - Complied", 
            "Group A (Treatment) - Non-Complied",
            "Group B (Training) - Complied", 
            "Group B (Training) - Non-Complied",
            "Group C (Control) - Complied", 
            "Group C (Control) - Non-Complied"
        ]
        
        return context

    # ----------------------------------
    # 0️⃣ Handle Direct CSV Download
    # ----------------------------------
    if 'download' in request.GET:
        filename = request.GET['download']
        if 'merged_df' not in session or 'kharif_df' not in session:
            return HttpResponse(b"No data loaded.", status=400)

        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split') if 'farm_daily_avg' in session else None
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split') if 'weekly_avg' in session else None

        downloadables = prepare_comprehensive_downloads(merged_df, kharif_df, farm_daily_avg, weekly_avg)
        if filename.startswith('farm_'):
            downloadables.update(get_per_farm_downloads(merged_df, farm_daily_avg))
        elif filename.startswith('village_'):
            downloadables.update(get_village_level_analysis(merged_df, kharif_df))
        if filename not in downloadables:
            return HttpResponse(b"File not found.", status=404)

        response = HttpResponse(downloadables[filename], content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename={filename}'
        return response

    # ----------------------------------
    # 1️⃣ Handle File Upload
    # ----------------------------------
    if request.method == 'POST' and 'water_file' in request.FILES:
        try:
            # Load the water level file
            kharif_df, water_df, missing_kharif, missing_water = load_and_validate_data(
                None, request.FILES['water_file']
            )
            if missing_water:
                context['error'] = f"Missing columns in water file: {missing_water}"
                return render(request, 'water_meters.html', context)

            # Try to load kharif data from Google Sheets first
            if 'kharif_file' not in request.FILES:  # Only use Google Sheets if no manual override
                print_status("Attempting to load master data from Google Sheets...", "process")
                try:
                    master_data = load_master_from_google_sheets()
                    if master_data and 'Farm details' in master_data:  # Use correct sheet name
                        kharif_df = master_data['Farm details']
                        print_status(f"Successfully loaded farm details from Google Sheets: {len(kharif_df)} rows", "success")
                    elif master_data:
                        # Try different potential sheet names
                        for sheet_name in master_data.keys():
                            if any(keyword in sheet_name.lower() for keyword in ['farm', 'kharif', 'detail']):
                                kharif_df = master_data[sheet_name]
                                print_status(f"Found farm data in sheet '{sheet_name}': {len(kharif_df)} rows", "success")
                                break
                        if kharif_df is None:
                            print_status(f"No farm details sheet found. Available sheets: {list(master_data.keys())}", "warning")
                except Exception as e:
                    print_status(f"Failed to load from Google Sheets: {e}", "error")

            # Fall back to manual upload if Google Sheets failed or manual override provided
            if kharif_df is None and 'kharif_file' in request.FILES:
                print_status("Loading farm details from manual upload...", "process")
                kharif_df, _, missing_kharif = load_and_validate_data(
                    request.FILES['kharif_file'], None
                )
                if missing_kharif:
                    context['error'] = f"Missing columns in kharif file: {missing_kharif}"
                    return render(request, 'water_meters.html', context)
            elif kharif_df is None:
                context['error'] = "Could not load farm details from Google Sheets and no manual file provided"
                return render(request, 'water_meters.html', context)

        except Exception as e:
            context['error'] = f"Error processing files: {e}"
            return render(request, 'water_meters.html', context)

        kharif_cleaned, water_cleaned = clean_and_process_data(kharif_df, water_df)
        
        # Prepare date range for filtering
        date_range = None
        if use_date_filter and start_date and end_date:
            date_range = (start_date, end_date)
        
        filtered_kharif, filtered_water = apply_comprehensive_filters(kharif_cleaned, water_cleaned, {
            'villages': [], 'date_range': date_range, 'available_groups': {},
            'remote_controllers': 'All', 'awd_study': 'All',
            'farming_method': 'All', 'min_readings': 1, 'remove_outliers': False
        })

        merged_df, farm_daily_avg, weekly_avg = create_merged_dataset(filtered_kharif, filtered_water)

        # Store in session
        session['merged_df'] = merged_df.to_json(orient='split')
        session['kharif_df'] = kharif_cleaned.to_json(orient='split')
        session['kharif_cleaned'] = kharif_cleaned.to_json(orient='split')  # Store original cleaned data
        session['water_cleaned'] = water_cleaned.to_json(orient='split')   # Store original cleaned data
        session['filtered_kharif'] = filtered_kharif.to_json(orient='split')
        session['filtered_water'] = filtered_water.to_json(orient='split')
        session['farm_daily_avg'] = farm_daily_avg.to_json(orient='split')
        session['weekly_avg'] = weekly_avg.to_json(orient='split')

        # Basic summary
        context['summary'] = {
            'total_records': len(merged_df),
            'unique_farms': merged_df['Farm ID'].nunique(),
            'total_villages': merged_df['Village'].nunique() if 'Village' in merged_df.columns else 0,
            'date_range': f"{merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}",
            'avg_level': round(merged_df['Water Level (mm)'].mean(), 1)
        }

        # Load full context using helper function
        context = load_full_context(context, merged_df, kharif_cleaned, farm_daily_avg, weekly_avg, session)

        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 2️⃣ Handle AJAX Farm Selection & Date Filter Updates
    # ----------------------------------
    if 'merged_df' in session and request.method == 'POST' and 'download_zip' not in request.POST and 'create_custom_group' not in request.POST and 'delete_custom_group' not in request.POST and 'update_date_filter' not in request.POST and 'update_village_comparison' not in request.POST and 'update_rc_comparison' not in request.POST and 'update_awd_comparison' not in request.POST:
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
        selected_farm = request.POST.get('selected_farm')
        if selected_farm:
            context['selected_farm'] = selected_farm
        
        # Load full context
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
        
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 2.5️⃣ Handle Date Filter Updates (requires data reprocessing)
    # ----------------------------------
    if 'merged_df' in session and request.method == 'POST' and 'update_date_filter' in request.POST:
        # Reload the original cleaned data
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        filtered_kharif = pd.read_json(session['filtered_kharif'], orient='split')
        filtered_water = pd.read_json(session['filtered_water'], orient='split')
        
        # Get the original cleaned data to reapply filters
        water_df = pd.read_json(session.get('water_cleaned', session['filtered_water']), orient='split')
        kharif_cleaned = pd.read_json(session.get('kharif_cleaned', session['kharif_df']), orient='split')
        
        # Prepare date range for filtering
        date_range = None
        if use_date_filter and start_date and end_date:
            date_range = (start_date, end_date)
        
        # Reapply filters with new date range
        filtered_kharif, filtered_water = apply_comprehensive_filters(kharif_cleaned, water_df, {
            'villages': [], 'date_range': date_range, 'available_groups': {},
            'remote_controllers': 'All', 'awd_study': 'All',
            'farming_method': 'All', 'min_readings': 1, 'remove_outliers': False
        })
        
        # Recreate merged dataset with filtered data
        merged_df, farm_daily_avg, weekly_avg = create_merged_dataset(filtered_kharif, filtered_water)
        
        # Update session with new filtered data
        session['merged_df'] = merged_df.to_json(orient='split')
        session['filtered_kharif'] = filtered_kharif.to_json(orient='split')
        session['filtered_water'] = filtered_water.to_json(orient='split')
        session['farm_daily_avg'] = farm_daily_avg.to_json(orient='split')
        session['weekly_avg'] = weekly_avg.to_json(orient='split')
        session.modified = True
        
        # Basic summary with filtered data
        context['summary'] = {
            'total_records': len(merged_df),
            'unique_farms': merged_df['Farm ID'].nunique(),
            'total_villages': merged_df['Village'].nunique() if 'Village' in merged_df.columns else 0,
            'date_range': f"{merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}",
            'avg_level': round(merged_df['Water Level (mm)'].mean(), 1)
        }
        
        # Load full context with filtered data
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
        
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 3️⃣ Download Full ZIP
    # ----------------------------------
    if request.method == 'POST' and 'download_zip' in request.POST:
        if 'merged_df' not in session:
            return HttpResponse(b"No data loaded.", status=400)
            
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        zip_buffer = create_zip_package(merged_df, kharif_df, farm_daily_avg, weekly_avg)
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=agri_dashboard_package.zip'
        return response

    # ----------------------------------
    # 4️⃣ Village Summary Download
    # ----------------------------------
    if 'download_village_summary' in request.GET:
        if 'merged_df' not in session:
            return HttpResponse(b"No data loaded.", status=400)
            
        merged_df = pd.read_json(session['merged_df'], orient='split')
        # Use selected villages from session if available
        selected_villages = session.get('selected_villages_comparison', request.GET.getlist('villages'))
        summary_csv = get_village_level_analysis(merged_df, selected_villages)['summary_df'].to_csv(index=False)
        return HttpResponse(summary_csv, content_type='text/csv', headers={
            'Content-Disposition': 'attachment; filename=village_summary.csv'
        })

    # ----------------------------------
    # 5️⃣ Village Analysis ZIP
    # ----------------------------------
    if 'download_village_analysis' in request.GET:
        if 'merged_df' not in session:
            return HttpResponse(b"No data loaded.", status=400)
            
        merged_df = pd.read_json(session['merged_df'], orient='split')
        # Use selected villages from session if available
        selected_villages = session.get('selected_villages_comparison', request.GET.getlist('villages'))
        village_result = get_village_level_analysis(merged_df, selected_villages)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('village_summary.csv', village_result['summary_df'].to_csv(index=False))
            if 'trend_df' in village_result:
                zf.writestr('village_daily_trends.csv', village_result['trend_df'].to_csv(index=False))
        zip_buffer.seek(0)
        return HttpResponse(zip_buffer.getvalue(), content_type='application/zip', headers={
            'Content-Disposition': 'attachment; filename=village_analysis.zip'
        })

    # ----------------------------------
    # 2.7️⃣ Handle Village Comparison Selection
    # ----------------------------------
    if request.method == 'POST':
        print(f"[DEBUG] POST request received. POST data keys: {list(request.POST.keys())}")
        print(f"[DEBUG] Checking for 'update_village_comparison' in POST: {'update_village_comparison' in request.POST}")
    
    if request.method == 'POST' and 'update_village_comparison' in request.POST:
        print(f"[DEBUG] Village comparison POST handler triggered")
        if 'merged_df' not in session:
            context['error'] = "No data loaded. Please upload files first."
            return render(request, 'water_meters.html', context)
        
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
        # Get selected villages from form
        selected_villages = request.POST.getlist('selected_villages_comparison')
        print(f"[DEBUG] Selected villages from form: {selected_villages}")
        
        # Store selected villages in session
        session['selected_villages_comparison'] = selected_villages
        session.modified = True
        
        # Load custom groups
        custom_groups = session.get('custom_groups', {})
        custom_groups_df = {}
        for k, v in custom_groups.items():
            try:
                df = pd.read_json(StringIO(v), orient='split')
                if not df.empty:
                    required = {"Farm ID", "Village", "Water Level (mm)", "Date", "Days from TPR"}
                    if required.issubset(df.columns):
                        custom_groups_df[k] = df
            except Exception as e:
                print(f"[ERROR] Failed to parse custom group '{k}':", e)
        
        # Generate comparative analysis with selected villages
        try:
            comparative_results = render_comparative_analysis(merged_df, kharif_df, custom_groups_df, selected_villages)
            
            # Check if village analysis was successful
            if "error" in comparative_results.get('village', {}):
                context['error'] = f"Village analysis error: {comparative_results['village']['error']}"
                context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
                return render(request, 'water_meters.html', context)
            
        except Exception as e:
            context['error'] = f"Error generating comparative analysis: {str(e)}"
            context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
            return render(request, 'water_meters.html', context)
        
        # Load basic context data without regenerating comparative analysis
        context['merged_loaded'] = True
        context['available_farms'] = sorted(merged_df["Farm ID"].unique())
        context['available_villages'] = sorted(merged_df["Village"].dropna().unique()) if 'Village' in merged_df.columns else []
        context['selected_farm'] = context.get('selected_farm') or (context['available_farms'][0] if context['available_farms'] else None)
        
        # Load custom groups
        custom_groups = session.get('custom_groups', {})
        custom_groups_df = {}
        for k, v in custom_groups.items():
            try:
                df = pd.read_json(StringIO(v), orient='split')
                if not df.empty:
                    required = {"Farm ID", "Village", "Water Level (mm)", "Date", "Days from TPR"}
                    if required.issubset(df.columns):
                        custom_groups_df[k] = df
            except Exception as e:
                print(f"[ERROR] Failed to parse custom group '{k}':", e)
        
        context['custom_groups'] = list(custom_groups_df.keys())
        
        # Use the comparative results we already generated (don't regenerate)
        context['comparative'] = {}
        
        # Village comparison with the new results
        if 'plot' in comparative_results['village']:
            context['comparative']['village'] = {
                'plot': comparative_results['village']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['village']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            }
        else:
            context['error'] = "Failed to generate village comparison plot"
            return render(request, 'water_meters.html', context)
        
        # Add other comparative results (regenerate only the non-village parts)
        context['comparative']['rc'] = {}
        context['comparative']['awd'] = {}
        context['comparative']['dsr_tpr'] = {}
        context['comparative']['compliance'] = {}
        context['comparative']['custom'] = {}
        
        if "error" not in comparative_results['rc']:
            context['comparative']['rc'] = {
                'plot': comparative_results['rc']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['rc']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            }
        
        if "error" not in comparative_results['awd']:
            context['comparative']['awd'] = {
                'plot': comparative_results['awd']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['awd']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            }
        
        if "error" not in comparative_results['dsr_tpr']:
            context['comparative']['dsr_tpr'] = {
                'plot': comparative_results['dsr_tpr']['plot'].to_html(full_html=False),
                'summary_df': comparative_results['dsr_tpr']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            }
        
        if "error" not in comparative_results['compliance']:
            context['comparative']['compliance'] = {
                'df': comparative_results['compliance']['df'].to_html(index=False, classes="table table-bordered table-sm"),
                'summary': comparative_results['compliance']['summary']
            }
        
        if comparative_results['custom']['chart_html']:
            context['comparative']['custom'] = {
                'chart_html': comparative_results['custom']['chart_html'],
                'summary_df': comparative_results['custom']['summary_df'].to_html(index=False, classes="table table-bordered table-sm")
            }
        
        # Add information about selection
        if selected_villages:
            context['village_selection_info'] = f"Showing {len(selected_villages)} selected villages out of {len(comparative_results['village'].get('all_villages', []))}"
        else:
            context['village_selection_info'] = f"Showing all {len(comparative_results['village'].get('all_villages', []))} villages"
        
        context['selected_villages_comparison'] = selected_villages
        
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 2.8️⃣ Handle Remote Controllers Comparison Selection
    # ----------------------------------
    if request.method == 'POST' and 'update_rc_comparison' in request.POST:
        print(f"[DEBUG] RC comparison POST handler triggered")
        if 'merged_df' not in session:
            context['error'] = "No data loaded. Please upload files first."
            return render(request, 'water_meters.html', context)
        
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
        # Get selected RC groups from form
        selected_rc_groups = request.POST.getlist('selected_rc_groups')
        print(f"[DEBUG] Selected RC groups from form: {selected_rc_groups}")
        
        # Store selected RC groups in session
        session['selected_rc_groups'] = selected_rc_groups
        session.modified = True
        
        # Load custom groups
        custom_groups = session.get('custom_groups', {})
        custom_groups_df = {}
        for k, v in custom_groups.items():
            try:
                df = pd.read_json(StringIO(v), orient='split')
                if not df.empty:
                    required = {"Farm ID", "Village", "Water Level (mm)", "Date", "Days from TPR"}
                    if required.issubset(df.columns):
                        custom_groups_df[k] = df
            except Exception as e:
                print(f"[ERROR] Failed to parse custom group '{k}':", e)
        
        # Generate comparative analysis with selected RC groups
        try:
            comparative_results = render_comparative_analysis(merged_df, kharif_df, custom_groups_df, selected_rc_groups=selected_rc_groups)
            
            # Check if RC analysis was successful
            if "error" in comparative_results.get('rc', {}):
                context['error'] = f"RC analysis error: {comparative_results['rc']['error']}"
                context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
                return render(request, 'water_meters.html', context)
            
        except Exception as e:
            context['error'] = f"Error generating RC comparative analysis: {str(e)}"
            context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
            return render(request, 'water_meters.html', context)
        
        # Load basic context and render with updated RC results
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session, comparative_results)
        context['selected_rc_groups'] = selected_rc_groups
        
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 2.9️⃣ Handle AWD Comparison Selection
    # ----------------------------------
    if request.method == 'POST' and 'update_awd_comparison' in request.POST:
        print(f"[DEBUG] AWD comparison POST handler triggered")
        if 'merged_df' not in session:
            context['error'] = "No data loaded. Please upload files first."
            return render(request, 'water_meters.html', context)
        
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
        # Get selected AWD groups from form
        selected_awd_groups = request.POST.getlist('selected_awd_groups')
        print(f"[DEBUG] Selected AWD groups from form: {selected_awd_groups}")
        
        # Store selected AWD groups in session
        session['selected_awd_groups'] = selected_awd_groups
        session.modified = True
        
        # Load custom groups
        custom_groups = session.get('custom_groups', {})
        custom_groups_df = {}
        for k, v in custom_groups.items():
            try:
                df = pd.read_json(StringIO(v), orient='split')
                if not df.empty:
                    required = {"Farm ID", "Village", "Water Level (mm)", "Date", "Days from TPR"}
                    if required.issubset(df.columns):
                        custom_groups_df[k] = df
            except Exception as e:
                print(f"[ERROR] Failed to parse custom group '{k}':", e)
        
        # Generate comparative analysis with selected AWD groups
        try:
            comparative_results = render_comparative_analysis(merged_df, kharif_df, custom_groups_df, selected_awd_groups=selected_awd_groups)
            
            # Check if AWD analysis was successful
            if "error" in comparative_results.get('awd', {}):
                context['error'] = f"AWD analysis error: {comparative_results['awd']['error']}"
                context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
                return render(request, 'water_meters.html', context)
            
        except Exception as e:
            context['error'] = f"Error generating AWD comparative analysis: {str(e)}"
            context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
            return render(request, 'water_meters.html', context)
        
        # Load basic context and render with updated AWD results
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session, comparative_results)
        context['selected_awd_groups'] = selected_awd_groups
        
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 6️⃣ Handle Custom Group Creation
    # ----------------------------------
    if request.method == 'POST' and 'create_custom_group' in request.POST:
        if 'merged_df' not in session:
            context['error'] = "No data loaded. Please upload files first."
            return render(request, 'water_meters.html', context)

        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')

        group_count = int(request.POST.get('group_count', 1))
        custom_groups = session.get('custom_groups', {})

        group_created = False
        for i in range(1, group_count + 1):
            group_name = request.POST.get(f'group_name_{i}', '').strip()
            selected_farms = request.POST.getlist(f'selected_farms_{i}')
            selected_villages = request.POST.getlist(f'selected_villages_{i}')
            
            if not group_name:
                context['error'] = f"Group name is required for group {i}."
                context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
                return render(request, 'water_meters.html', context)
                
            if not selected_farms and not selected_villages:
                context['error'] = f"Please select at least one farm or village for group {i}."
                context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
                return render(request, 'water_meters.html', context)
                
            # Create custom group DataFrame
            custom_group_data = merged_df.copy()
            if selected_farms:
                custom_group_data = custom_group_data[custom_group_data['Farm ID'].isin(selected_farms)]
            if selected_villages:
                custom_group_data = custom_group_data[custom_group_data['Village'].isin(selected_villages)]
                
            if custom_group_data.empty:
                context['error'] = f"No data found for the selected farms/villages in group {i}."
                context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
                return render(request, 'water_meters.html', context)
                
            custom_groups[group_name] = custom_group_data.to_json(orient='split')
            group_created = True

        if group_created:
            session['custom_groups'] = custom_groups
            session.modified = True
            context['success_message'] = f"Custom group(s) created successfully."

        # Load full context
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 7️⃣ Handle Custom Group Deletion
    # ----------------------------------
    if request.method == 'POST' and 'delete_custom_group' in request.POST:
        if 'merged_df' not in session:
            context['error'] = "No data loaded. Please upload files first."
            return render(request, 'water_meters.html', context)
        
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
        group_name = request.POST.get('group_name', '').strip()
        if not group_name:
            context['error'] = "Group name is required for deletion."
            context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
            return render(request, 'water_meters.html', context)
        
        custom_groups = session.get('custom_groups', {})
        if group_name in custom_groups:
            del custom_groups[group_name]
            session['custom_groups'] = custom_groups
            session.modified = True
            context['success_message'] = f"Custom group '{group_name}' deleted successfully."
        else:
            context['error'] = f"Custom group '{group_name}' not found."
        
        # Load full context
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)
        return render(request, 'water_meters.html', context)

    # ----------------------------------
    # 8️⃣ Handle Comparative Analysis Downloads
    # ----------------------------------
    if request.method == 'GET' and any(k in request.GET for k in [
        'download_village_analysis', 'download_rc_analysis', 'download_awd_analysis', 
        'download_dsr_tpr_analysis', 'download_custom_group_analysis'
    ]):
        if 'merged_df' not in session:
            return HttpResponse(b"No data loaded.", status=400)
            
        section_map = {
            'download_village_analysis': 'village',
            'download_rc_analysis': 'rc',
            'download_awd_analysis': 'awd',
            'download_dsr_tpr_analysis': 'dsr_tpr',
            'download_custom_group_analysis': 'custom'
        }
        
        for key, section in section_map.items():
            if key in request.GET:
                merged_df = pd.read_json(session['merged_df'], orient='split')
                kharif_df = pd.read_json(session['kharif_df'], orient='split')
                custom_groups = session.get('custom_groups', {})
                custom_groups_df = {k: pd.read_json(StringIO(v), orient='split') for k, v in custom_groups.items()}
                
                summary_df = get_comparative_analysis(section, merged_df, kharif_df, custom_groups_df)
                if summary_df is not None:
                    summary_csv = summary_df.to_csv(index=False)
                    return HttpResponse(summary_csv, content_type='text/csv', headers={
                        'Content-Disposition': f'attachment; filename={section}_analysis.csv'
                    })
                else:
                    return HttpResponse(b"No data available.", status=400)

    # ----------------------------------
    # 9️⃣ Handle Data Reload for Existing Sessions
    # ----------------------------------
    if 'merged_df' in session and request.method == 'GET':
        merged_df = pd.read_json(session['merged_df'], orient='split')
        kharif_df = pd.read_json(session['kharif_df'], orient='split')
        farm_daily_avg = pd.read_json(session['farm_daily_avg'], orient='split')
        weekly_avg = pd.read_json(session['weekly_avg'], orient='split')
        
        # Load full context for GET requests when data exists
        context = load_full_context(context, merged_df, kharif_df, farm_daily_avg, weekly_avg, session)

    # ----------------------------------
    # 🖼️ Final Page Render
    # ----------------------------------
    return render(request, 'water_meters.html', context)

# Helper for pretty download names
def pretty_download_name(filename):
    mapping = {
        "01_complete_merged_data.csv": "Complete Merged Dataset",
        "02_raw_kharif_data.csv": "Raw Kharif Data",
        "20_village_summary.csv": "Village Summary",
        "21_farm_summary.csv": "Farm Summary",
        "30_remote_controller_summary.csv": "Remote Controller Summary",
        "31_awd_study_summary.csv": "AWD Study Summary",
        "32_dsr_vs_tpr_summary.csv": "DSR vs TPR Summary",
    }
    # For study group files
    if filename.endswith(".csv") and filename[3:-4]:
        name = filename[3:-4].replace("_", " ").title()
        if "Remote Controllers Treatment" in name:
            return "Remote Controllers Treatment"
        if "Remote Controllers Control" in name:
            return "Remote Controllers Control"
        if "Awd Group A Treatment" in name:
            return "AWD Group A Treatment"
        if "Awd Group B Training" in name:
            return "AWD Group B Training"
        if "Awd Group C Control" in name:
            return "AWD Group C Control"
        if "Dsr Farms" in name:
            return "DSR Farms"
        if "Tpr Farms" in name:
            return "TPR Farms"
        return name
    return mapping.get(filename, filename)



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

from .utils import kharif2025_farms, get_2025plots, get_meters_by_village, apply_7day_sma, create_weekly_delta, get_acreage, create_delta_vs_days_from_tpr, generate_delta_vs_days_groupwise_plots, get_dates_table, get_days_reading_table



# Replace the existing meter_reading_25_view function with this updated version:
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
            start_date_for_readings = pd.to_datetime(date_range_info['current_min'])
            end_date_for_readings = pd.to_datetime(date_range_info['current_max'])
            if download_request == 'dates_reading':
                combined_df = get_dates_table(raw_df, master25, farm_dict, start_date_for_readings, end_date_for_readings, download_request)
            elif download_request == 'days_reading':
                if use_date_filter and filter_start_date and filter_end_date:
                    filter_start = pd.to_datetime(filter_start_date)
                    filter_end = pd.to_datetime(filter_end_date)
                    combined_df = get_days_reading_table(raw_df, master25, farm_dict, start_date_enter=filter_start, end_date_enter=filter_end)
                else:
                    combined_df = get_days_reading_table(raw_df, master25, farm_dict)

            else:
                cols_vs_request = {"days_daily_avg": "m³ per Acre per Avg Day", "days_delta_acre": "m³ per Acre", "days_delta":"Delta m³", "dates_daily_avg": "m³ per Acre per Avg Day", "dates_delta_acre": "m³ per Acre", "dates_delta":"Delta m³"}
                days_or_date = "Day" if download_request.startswith("days") else "Date"
                col_to_get = cols_vs_request[download_request]
                if use_date_filter and filter_start_date and filter_end_date:
                    filter_start = pd.to_datetime(filter_start_date)
                    filter_end = pd.to_datetime(filter_end_date)
                    combined_df = get_tables(raw_df, master25, farm_dict, col_to_get, days_or_date, start_date_for_readings, end_date_for_readings,start_date_enter=filter_start, end_date_enter=filter_end)
                else:
                    combined_df = get_tables(raw_df, master25, farm_dict, col_to_get, days_or_date, start_date_for_readings, end_date_for_readings)

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
        # Reconstruct results based on what was selected - GENERATE MATPLOTLIB FOR REPORTS
        filter_type = None
        filter_value = None
        report_results = []
        
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
                report_results.append(block)
                
        elif request.POST.get('report_filter_type') == 'village':
            selected_village = request.POST.get('report_filter_value')
            filter_type = "Village Filter"
            filter_value = selected_village
            from .utils import get_meters_by_village
            
            village_meters = get_meters_by_village(raw_df, selected_village)
            farm_dict = kharif2025_farms(master25)
            meter_to_farm = {}
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
                        report_results.append(block)
        
        # Generate Word report
        from .utils import generate_word_report
        docx_buffer = generate_word_report(report_results, filter_type, filter_value, raw_df, master25)
        
        # Return Word document
        response = HttpResponse(
            docx_buffer.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        response['Content-Disposition'] = f'attachment; filename="water_analysis_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.docx"'
        return response

    # 6) When a farm is selected, generate BOTH matplotlib (for reports) and plotly (for display)
    if selected and raw_df is not None and master25:
        mapping = kharif2025_farms(master25)
        meters = mapping.get(selected, [])
        
        # Generate PLOTLY plots for portal display
        if use_date_filter and filter_start_date and filter_end_date:
            filter_start = pd.to_datetime(filter_start_date)
            filter_end = pd.to_datetime(filter_end_date)
            plotly_htmls = get_2025plots_plotly(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
            # Also generate matplotlib for potential reports
            encoded_imgs = get_2025plots(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
        else:
            plotly_htmls = get_2025plots_plotly(raw_df, master25, selected, meters)
            # Also generate matplotlib for potential reports
            encoded_imgs = get_2025plots(raw_df, master25, selected, meters)
        
        acreage_of_selected = get_acreage(master25, selected)


        # Group plots per meter - 4 plotly plots per meter
        for idx, meter in enumerate(meters):
            block = {
                'meter': meter,
                'plotly_plots': plotly_htmls[4*idx : 4*idx + 4],  # Plotly for display
                'plots': encoded_imgs[4*idx : 4*idx + 4],         # Matplotlib for reports
                'is_combined': False,
                'farm': selected,
                'acre': acreage_of_selected
            }
            results.append(block)
        
        # Generate combined plots if multiple meters
        if len(meters) > 1:
            if use_date_filter and filter_start_date and filter_end_date:
                filter_start = pd.to_datetime(filter_start_date)
                filter_end = pd.to_datetime(filter_end_date)
                combined_plotly = get_2025plots_combined_plotly(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
                combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters, start_date_enter=filter_start, end_date_enter=filter_end)
            else:
                combined_plotly = get_2025plots_combined_plotly(raw_df, master25, selected, meters)
                combined_imgs = get_2025plots_combined(raw_df, master25, selected, meters)
            

            if combined_plotly:
                combined_block = {
                    'meter': ' + '.join(meters),
                    'plotly_plots': combined_plotly,  # Plotly for display
                    'plots': combined_imgs,           # Matplotlib for reports
                    'is_combined': True,
                    'farm': selected,
                    'acre': acreage_of_selected
                }
                results.append(combined_block)
    
    # 7) When a village is selected, generate graphs for all meters in that village
    elif selected_village and raw_df is not None and master25:
        from .utils import get_meters_by_village
        
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
            acreage_of_selected = get_acreage(master25, farm_id)
            # Individual meter plots
            for meter in farm_meters:
                if use_date_filter and filter_start_date and filter_end_date:
                    filter_start = pd.to_datetime(filter_start_date)
                    filter_end = pd.to_datetime(filter_end_date)
                    meter_plotly = get_2025plots_plotly(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
                    meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter], start_date_enter=filter_start, end_date_enter=filter_end)
                else:
                    meter_plotly = get_2025plots_plotly(raw_df, master25, farm_id, [meter])
                    meter_imgs = get_2025plots(raw_df, master25, farm_id, [meter])
                
                
                for idx in range(0, len(meter_plotly), 4):
                    block = {
                        'meter': meter,
                        'farm': farm_id,
                        'plotly_plots': meter_plotly[idx:idx+4],  # Plotly for display
                        'plots': meter_imgs[idx:idx+4],           # Matplotlib for reports
                        'is_combined': False,
                        'acre': acreage_of_selected
                    }
                    results.append(block)
            
            # Combined plots if multiple meters for this farm
            if len(farm_meters) > 1:
                if use_date_filter and filter_start_date and filter_end_date:
                    filter_start = pd.to_datetime(filter_start_date)
                    filter_end = pd.to_datetime(filter_end_date)
                    combined_plotly = get_2025plots_combined_plotly(raw_df, master25, farm_id, farm_meters, start_date_enter=filter_start, end_date_enter=filter_end)
                    combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters, start_date_enter=filter_start, end_date_enter=filter_end)
                else:
                    combined_plotly = get_2025plots_combined_plotly(raw_df, master25, farm_id, farm_meters)
                    combined_imgs = get_2025plots_combined(raw_df, master25, farm_id, farm_meters)
                
                if combined_plotly:
                    combined_block = {
                        'meter': ' + '.join(farm_meters),
                        'farm': farm_id,
                        'plotly_plots': combined_plotly,  # Plotly for display
                        'plots': combined_imgs,           # Matplotlib for reports
                        'is_combined': True,
                        'acre': acreage_of_selected
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
        selected_checkboxes = request.POST.getlist('group_category')

        raw_df = pd.read_json(request.session['raw25']) if 'raw25' in request.session else None
        master25 = {
            name: pd.read_json(json_str)
            for name, json_str in request.session.get('master25', {}).items()
        } if 'master25' in request.session else None
        
        # Apply date filter if enabled
        start_date = request.session.get('filter_start_date')
        end_date = request.session.get('filter_end_date')
        use_filter = request.session.get('use_date_filter', False)

        # Handle report download
        if request.POST.get('download_group_report') and 'group_plot' in request.session:
            from .utils import generate_group_analysis_report
            stored_data = request.session.get('group_analysis_data', {})

            docx_buffer = generate_group_analysis_report(
                stored_data.get('group_type', 'Combined'),
                stored_data.get('selected_groups', []),
                request.session.get('group_plot', ''),
                request.session.get('group_plot2', ''),
                stored_data.get('group_farms', {})
            )

            response = HttpResponse(
                docx_buffer.getvalue(),
                content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
            response['Content-Disposition'] = f'attachment; filename="group_analysis_combined_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.docx"'
            return response

        if selected_checkboxes and raw_df is not None and master25:
            kharif_df = master25['Farm details']
            group_farms_dict = {}
            group_farms_len = {}

            # Unified column mapping
            column_map = {
                "Remote Group-A Complied": "Kharif 25 - Remote Controllers Study - Group A - Treatment - complied (Y/N)",
                "Remote Group-A Non-Complied": "Kharif 25 - Remote Controllers Study - Group A - Treatment - NON-complied (Y/N)",
                "Remote Group-A Whole": [
                    "Kharif 25 - Remote Controllers Study - Group A - Treatment - complied (Y/N)",
                    "Kharif 25 - Remote Controllers Study - Group A - Treatment - NON-complied (Y/N)"
                ],
                "Remote Group-B Complied": "Kharif 25 - Remote Controllers Study - Group B - Control - complied (Y/N)",
                "Remote Group-B Non-Complied": "Kharif 25 - Remote Controllers Study - Group B - Control - NON-complied (Y/N)",
                "Remote Group-B Whole": [
                    "Kharif 25 - Remote Controllers Study - Group B - Control - complied (Y/N)",
                    "Kharif 25 - Remote Controllers Study - Group B - Control - NON-complied (Y/N)"
                ],
                "AWD Group-A Complied": "Kharif 25 - AWD Study - Group A - Treatment - complied (Y/N)",
                "AWD Group-A Non-Complied": "Kharif 25 - AWD Study - Group A - Treatment - Non-complied (Y/N)",
                "AWD Group-A Whole": [
                    "Kharif 25 - AWD Study - Group A - Treatment - complied (Y/N)",
                    "Kharif 25 - AWD Study - Group A - Treatment - Non-complied (Y/N)"
                ],
                "AWD Group-B Complied": "Kharif 25 - AWD Study - Group B - Complied (Y/N)",
                "AWD Group-B Non-Complied": "Kharif 25 - AWD Study - Group B - Non-complied (Y/N)",
                "AWD Group-B Whole": [
                    "Kharif 25 - AWD Study - Group B - Complied (Y/N)",
                    "Kharif 25 - AWD Study - Group B - Non-complied (Y/N)"
                ],
                "AWD Group-C Complied": "Kharif 25 - AWD Study - Group C - Complied (Y/N)",
                "AWD Group-C Non-Complied": "Kharif 25 - AWD Study - Group C - non-complied (Y/N)",
                "AWD Group-C Whole": [
                    "Kharif 25 - AWD Study - Group C - Complied (Y/N)",
                    "Kharif 25 - AWD Study - Group C - non-complied (Y/N)"
                ],
                "TPR": "Kharif 25 - TPR Group Study (Y/N)",
                "DSR": "Kharif 25 - DSR farm Study (Y/N)"
            }

            for label in selected_checkboxes:
                cols = column_map[label]
                if isinstance(cols, str):
                    condition = kharif_df[cols].fillna(0) == 1
                else:  # it's a list (Whole group)
                    condition = kharif_df[cols[0]].fillna(0) == 1
                    for c in cols[1:]:
                        condition |= kharif_df[c].fillna(0) == 1

                farm_ids = kharif_df.loc[condition, "Kharif 25 Farm ID"].tolist()
                group_farms_dict[label] = farm_ids
                group_farms_len[label] = len(farm_ids)

            group_dfs, group_dfs2, group_dfs4 = [], [], []
            group_delta_vs_days_data = {}  # Store delta vs days data for each group

            for label, farms in group_farms_dict.items():
                if use_filter:
                    filter_start = pd.to_datetime(start_date)
                    filter_end = pd.to_datetime(end_date)
                    df = calculate_avg_m3_per_acre("Combined", label, farms, raw_df, master25, 'm³ per Acre per Avg Day', start_date_enter=filter_start, end_date_enter=filter_end)
                    df2 = create_weekly_delta("Combined", label, farms, raw_df, master25, "Delta m³", start_date_enter=filter_start, end_date_enter=filter_end)
                    df4 = create_weekly_delta("Combined", label, farms, raw_df, master25, "m³ per Acre", start_date_enter=filter_start, end_date_enter=filter_end)
                    # Get delta vs days from TPR data for this group
                    delta_vs_days_df = create_delta_vs_days_from_tpr("Combined", label, farms, raw_df, master25, start_date_enter=filter_start, end_date_enter=filter_end)
                else:
                    df = calculate_avg_m3_per_acre("Combined", label, farms, raw_df, master25, 'm³ per Acre per Avg Day')
                    df2 = create_weekly_delta("Combined", label, farms, raw_df, master25, "Delta m³")
                    df4 = create_weekly_delta("Combined", label, farms, raw_df, master25, "m³ per Acre")
                    # Get delta vs days from TPR data for this group
                    delta_vs_days_df = create_delta_vs_days_from_tpr("Combined", label, farms, raw_df, master25)
                
                group_dfs.append(df)
                group_dfs2.append(df2)
                group_dfs4.append(df4)
                group_delta_vs_days_data[label] = delta_vs_days_df

            final_df = group_dfs[0]
            for df in group_dfs[1:]:
                final_df = pd.merge(final_df, df, on="Day", how="outer")

            sma_df = apply_7day_sma(final_df)

            group_plot3 = generate_group_analysis_plot(sma_df, "7 day SMA for daily average", group_farms_len, "Days")
            request.session['group_plot3'] = group_plot3

            
            group_plot = generate_group_analysis_plot(final_df, "Daily Average m3/acre", group_farms_len, "Days")
            request.session['group_plot'] = group_plot

            final_df2 = group_dfs2[0]
            for df in group_dfs2[1:]:
                final_df2 = pd.merge(final_df2, df, on="Weeks", how="outer")
            group_plot2 = generate_group_analysis_plot(final_df2, "Delta m3", group_farms_len, "Weeks")
            request.session['group_plot2'] = group_plot2

            final_df4 = group_dfs4[0]
            for df in group_dfs2[1:]:
                final_df4 = pd.merge(final_df4, df, on="Weeks", how="outer")
            group_plot4 = generate_group_analysis_plot(final_df4, "Delta m3/acre", group_farms_len, "Weeks")
            request.session['group_plot4'] = group_plot4

            # Generate the new delta vs days from TPR plots (one per group)
            from .utils import generate_delta_vs_days_groupwise_plots
            groupwise_plots = generate_delta_vs_days_groupwise_plots(group_delta_vs_days_data)

            request.session['group_analysis_data'] = {
                'group_type': 'Combined',
                'selected_groups': selected_checkboxes,
                'group_farms': group_farms_dict
            }

            return render(request, 'grouping.html', {
                'group_plot': group_plot,
                'group_plot2': group_plot2,
                'group_plot3': group_plot3,
                'group_plot4': group_plot4,
                'output': True,
                'group_type': 'Combined',
                'selected_groups': selected_checkboxes,
                'groupwise_plots': groupwise_plots,
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


def agriculture_view(request):
    return render(request, 'agriculture.html')

def crop_view(request):
    return render(request, 'crop_residue.html')

def dsr_view(request):
    return render(request, 'dsr.html')

def farmers_view(request):
    return render(request, 'farmers.html')

def stages_view(request):
    return render(request, 'stages.html')

def tpr_view(request):
    return render(request, 'tpr.html')

def tubewell_view(request):
    return render(request, 'tubewell.html')

def farmer_engagement(request):
    return render(request, 'farmer-engagement.html')
