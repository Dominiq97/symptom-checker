# django-symptom-checker-app

A tutorial app to explain django dynamic formsets. Blog post can be found [here](https://medium.com/@taranjeet/adding-forms-dynamically-to-a-django-formset-375f1090c2b0)

### Setup

* Clone the project

```
git clone https://github.com/taranjeet/django-checker-app.git
```

* Make sure python and virtual environment is installed. Create virtual environment and install packages

```
cd django-checker-app
virtualenv -p $(which python3) pyenv
# activate virtual environment
source pyenv/bin/activate
# install packages
pip install -r requirements.txt
cd djchecker
```

* Run migrate command

```
python manage.py migrate
```

* Run the server

```
python manage.py runserver
```

### How to use

The project contains three routes namely

* /store/symptom/create_normal

This is the demo for adding multiple symptoms using normal forms

* /store/symptom/create_model

This is the demo for adding multiple symptoms using model forms

* /store/symptom/create_with_author

This is the demo for adding a symptom with multiple author, both using model forms
