from django.db import models


class VideoAnalysis(models.Model):

    video = models.FileField(
        upload_to="videos/"
    )

    uploaded_at = models.DateTimeField(
        auto_now_add=True
    )

    prediction = models.CharField(
        max_length=20,
        blank=True,
        null=True
    )

    confidence = models.FloatField(
        blank=True,
        null=True
    )

    frames_analyzed = models.IntegerField(
        default=0
    )

    processing_time = models.FloatField(
        blank=True,
        null=True
    )

    model_version = models.CharField(
        max_length=100,
        blank=True,
        null=True
    )

    def __str__(self):

        return f"{self.video.name} - {self.prediction}"