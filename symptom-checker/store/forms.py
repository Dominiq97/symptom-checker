from django import forms
from django.forms import (formset_factory, modelformset_factory)

from .models import (Symptom, Author)


class SymptomForm(forms.Form):
    name = forms.CharField(
        label='Symptom Name',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Symptom Name here'
        })
    )


class SymptomModelForm(forms.ModelForm):

    class Meta:
        model = Symptom
        fields = ('name', )
        labels = {
            'name': 'Symptom Name'
        }
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter Symptom Name here'
                }
            )
        }


SymptomFormset = formset_factory(SymptomForm)
SymptomModelFormset = modelformset_factory(
    Symptom,
    fields=('name', ),
    extra=1,
    widgets={
        'name': forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Symptom Name here'
            }
        )
    }
)

AuthorFormset = modelformset_factory(
    Author,
    fields=('name', ),
    extra=1,
    widgets={'name': forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Author Name here'
        })
    }
)
