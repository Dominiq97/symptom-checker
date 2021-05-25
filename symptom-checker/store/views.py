from django.shortcuts import render, redirect
from django.views import generic

from .forms import (
    SymptomFormset,
    SymptomModelFormset,
    SymptomModelForm,
    AuthorFormset
)
from .models import Symptom, Author


def create_symptom_normal(request):
    template_name = 'store/create_normal.html'
    heading_message = 'Formset Demo'
    if request.method == 'GET':
        formset = SymptomFormset(request.GET or None)
    elif request.method == 'POST':
        formset = SymptomFormset(request.POST)
        if formset.is_valid():
            for form in formset:
                name = form.cleaned_data.get('name')
                if name:
                    Symptom(name=name).save()
            print(formset.cleaned_data)
            return redirect('store:symptom_list')

    return render(request, template_name, {
        'formset': formset,
        'heading': heading_message,
    })


class SymptomListView(generic.ListView):

    model = Symptom
    context_object_name = 'symptoms'
    template_name = 'store/list.html'


def create_symptom_model_form(request):
    template_name = 'store/create_normal.html'
    heading_message = 'Model Formset Demo'
    if request.method == 'GET':
        formset = SymptomModelFormset(queryset=Symptom.objects.none())
    elif request.method == 'POST':
        formset = SymptomModelFormset(request.POST)
        if formset.is_valid():
            for form in formset:
                # only save if name is present
                if form.cleaned_data.get('name'):
                    form.save()
            return redirect('store:symptom_list')

    return render(request, template_name, {
        'formset': formset,
        'heading': heading_message,
    })


def create_symptom_with_authors(request):
    template_name = 'store/create_with_author.html'
    if request.method == 'GET':
        symptomform = SymptomModelForm(request.GET or None)
        formset = AuthorFormset(queryset=Author.objects.none())
    elif request.method == 'POST':
        symptomform = SymptomModelForm(request.POST)
        formset = AuthorFormset(request.POST)
        if symptomform.is_valid() and formset.is_valid():
            # first save this symptom, as its reference will be used in `Author`
            symptom = symptomform.save()
            for form in formset:
                # so that `symptom` instance can be attached.
                author = form.save(commit=False)
                author.symptom = symptom
                author.save()
            return redirect('store:symptom_list')
    return render(request, template_name, {
        'symptomform': symptomform,
        'formset': formset,
    })
