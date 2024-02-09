from django.urls import path
from . import views

app_name = "maizeai"

urlpatterns = [
    path("upload-image/", views.image_upload_view, name="upload_image"),
    path("image_upload_view/", views.image_upload_view, name="upload_image"),
    path("upload_imageMaizeDisease/", views.upload_imageMaizeDisease, name="upload_imageMaizeDisease"),
    path("get_user_results/", views.get_user_results, name="get_user_results"),
]
