from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_stream, name='train_stream'),
    path('load/', views.load_results, name='load_results'),
    path('wk11/train/', views.train_stream, name='train_stream_wk11'),
    path('wk11/load/', views.load_results, name='load_results_wk11'),
]