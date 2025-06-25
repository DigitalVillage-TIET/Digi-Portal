"""
URL configuration for digivi project.

The urlpatterns list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from django.views.generic.base import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('api/token/', views.api_token, name='api_token'),
    path('', views.landing, name='landing'),
    path('index/', views.index, name='index'),
    path('meter-reading-25/', views.meter_reading_25_view, name='meter_reading_25'),
    path('meter_reading/', views.meter_reading, name='meter_reading'),
    path('water_level/', views.water_level, name='water_level'),
    path('farmer_survey/', views.farmer_survey, name='farmer_survey'),
    path('evapotranspiration/', views.evapotranspiration, name='evapotranspiration'),
    path('mapping/', views.mapping, name='mapping'),
    path('grouping-25/', views.grouping_25, name='grouping-25'),
    path('home/', views.landing_protected, name='landing_protected'),
]

