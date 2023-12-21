from django.urls import path
# from .views import upload_image
from . import views

urlpatterns = [
    #path('upload/', views.upload_image, name='upload_image'),
    #path('process_images/', views.process_images, name='process_images'),
    path('image_upload_view/', views.image_upload_view, name='image_upload_view'),
]
