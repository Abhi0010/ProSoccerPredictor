from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='soccer-home'),
    path('about/', views.about, name='soccer-about'),
    path('contact/', views.contact, name='soccer-contact'),
    path('login/', views.login, name='soccer-login'),
    path('register/', views.register, name='soccer-register'),
    path('myprofile/', views.myprofile, name='soccer-myprofile'),
    path('prediction/', views.prediction, name='soccer-prediction'),
    path('analysis/', views.analysis, name='soccer-analysis'),
    path('predictor/', views.predictor, name='soccer-predictor')

]
