
from django.urls import path
from . import views


# Task 4: Add the URL of the documentation_interface view
# Task 11: Add the URL of the generate_documentation view
urlpatterns = [
    path('', views.github_input_url_view, name='github_input_url_view'),
    path('generate/', views.generate_documentation_view, name='generate_documentation_view'),
]