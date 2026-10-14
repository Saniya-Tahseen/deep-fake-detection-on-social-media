from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path("index.html", views.index, name="index_html"),
    path("UserLogin.html", views.UserLogin, name="UserLogin"),
    path("UserLogin", views.UserLogin, name="UserLoginSimple"),
    path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
    path("LoadDataset", views.LoadDataset, name="LoadDataset"),
    path("FastText", views.FastText, name="FastText"),
    path("TrainML", views.TrainML, name="TrainML"),
    path("DetectFake.html", views.DetectFake, name="DetectFake"),
    path("DetectFake", views.DetectFake, name="DetectFakeSimple"),
    path("DetectFakeAction", views.DetectFakeAction, name="DetectFakeAction"),
]
