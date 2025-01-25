from django.urls import path
from .views import QAView, HomeView

urlpatterns = [
    path('ask/', QAView.as_view(), name='ask'),
    path('', HomeView.as_view(), name='home'),
]