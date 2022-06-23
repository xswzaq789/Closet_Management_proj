import os

from django.db import models
from django.utils.translation import gettext_lazy as _
from config.models import CreationModificationDateBase

# https://proglish.tistory.com/53
# https://devkor.tistory.com/entry/02-Django-Rest-Framework-%EA%B0%9C%EB%B0%9C-%ED%99%98%EA%B2%BD-%EC%84%B8%ED%8C%85?category=734691

# https://dev-yakuza.posstree.com/ko/django/models/


class ImageModel(CreationModificationDateBase):
    image = models.ImageField(_("image"), upload_to='images')
    class Meta:
        verbose_name = "Image"
        verbose_name_plural = "Images"

    def __str__(self):
        return str(os.path.split(self.image.path)[-1])
