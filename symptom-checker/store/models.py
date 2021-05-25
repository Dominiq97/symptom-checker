from django.db import models


class Symptom(models.Model):

    name = models.CharField(max_length=255)

    class Meta:
        db_table = 'symptom'

    def __str__(self):
        return self.name

    def get_authors(self):
        return ', '.join(self.authors.all().values_list('name', flat=True))


class Author(models.Model):

    name = models.CharField(max_length=255)
    symptom = models.ForeignKey(
        Symptom,
        related_name='authors', on_delete=models.SET_NULL,
        null=True)

    class Meta:
        db_table = 'author'

    def __str__(self):
        return self.name
