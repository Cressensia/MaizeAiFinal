from django.urls import path
# from .views import upload_image
from . import views

urlpatterns = [
    path('image_upload_view/', views.image_upload_view, name='image_upload_view'),
]
