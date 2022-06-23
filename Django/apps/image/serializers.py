from rest_framework import serializers
from .models import ImageModel

class ImageSerializer(serializers.Serializer):
    url = serializers.CharField(max_length=100, default='')
    clothes_type = serializers.CharField(max_length=30, default='')

    def create(self, validated_data):
        return ImageModel.objects.create(validated_data)
    def update(self, instance, validated_data):
        instance.url = validated_data.get('url', instance.url)
        instance.clothes_type = validated_data.get('clothes_type', instance.clothes_type)
        return instance