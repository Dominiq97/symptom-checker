from django.urls import re_path

from .views import (
    create_symptom_normal,
    create_symptom_model_form,
    create_symptom_with_authors,
    result,
    SymptomListView,
)

app_name = 'store'

urlpatterns = [

    re_path(r'^symptom/create_normal', create_symptom_normal, name='create_symptom_normal'),
    re_path(r'^symptom/create_model', create_symptom_model_form, name='create_symptom_model_form'),
    re_path(r'^symptom/create_with_author', create_symptom_with_authors, name='create_symptom_with_authors'),
    re_path(r'^symptom/result', result, name='result'),
    re_path(r'^symptom/list', SymptomListView.as_view(), name='symptom_list'),

]
