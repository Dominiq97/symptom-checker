from django.db import models


class Symptom(models.Model):

    name = models.CharField(max_length=255)

    class Meta:
        db_table = 'symptom'

    def __str__(self):
        return self.name



