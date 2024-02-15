from django.urls import path
# from .views import upload_image
from . import views

urlpatterns = [
    path('image_upload_view/', views.image_upload_view, name='image_upload_view'),
    path('save_user/', views.save_user, name='save_user'),
    path('manage_user/', views.manage_user, name='manage_user'),
    path('get_results_by_email/', views.get_results_by_email, name='get_results_by_email'),
    path('delete_record/<str:collection_name>/<str:document_id>/', views.delete_record, name='delete_record'),
    path('update_plots/', views.update_plots, name='update_plots'),
    path("upload_imageMaizeDisease/", views.upload_imageMaizeDisease, name="upload_imageMaizeDisease"),
    path('get_total_count/', views.get_total_count, name='get_total_count'),
    path('get_monthly_count/', views.get_monthly_count, name='get_monthly_count'),
]
