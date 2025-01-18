from django.db import models

# Create your models here.
class Member (models.Model):
    username = models.CharField(max_length=25, unique=True)
    passward = models.CharField(max_length=25)
    
    def __str__(self):
        return f"{self.username} {self.passward}"