from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('getDataFromLLM/',views.getDataFromOpenAIAPI,name='openai'),
    path('uploadFile/',views.uploadFile,name='upload file'),
    path('getEmbedding/',views.getEmbedding,name='text embedding')
]
