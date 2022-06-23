from django.urls import path, include
from . import views


app_name = "image"

urlpatterns = [
    path("", views.UploadImage.as_view(), name="upload_image_url"),
    path('api-auth/', include('rest_framework.urls'))
]
