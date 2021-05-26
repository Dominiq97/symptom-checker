from django.urls import re_path

from .views import (
    create_diagnosis,
    result,
    SymptomListView,
)

app_name = 'store'

urlpatterns = [

    re_path(r'^symptom/create_diagnosis', create_diagnosis, name='create_diagnosis'),
    re_path(r'^symptom/result', result, name='result'),
    re_path(r'^symptom/list', SymptomListView.as_view(), name='symptom_list'),

]
