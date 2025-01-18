from django.contrib import admin
from django.urls import path
from . import views
from .views import login_view, logout_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('members/', views.members, name='members'),
    path('homee/', views.homee, name='homee'),
    path('about/', views.about, name='about'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('signup/', views.sign_up_view, name='signup'),
    path('calculator/', views.calculator_view, name='calculator'),
    path('', views.homee, name='homee'),
    path('sm/', views.trigger_virtual_mouse, name='sm'),
    path('vm/', views.trigger_volume_control, name='vm'),
    path('stop_volume_control/', views.stop_volume_control, name='stop_volume_control'),
    path('virtual-drawing/', views.virtual_drawing, name='virtual_drawing'),







]
