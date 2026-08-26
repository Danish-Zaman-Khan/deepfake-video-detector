from django import forms

from .models import VideoAnalysis


class VideoUploadForm(forms.ModelForm):

    class Meta:

        model = VideoAnalysis

        fields = [
            "video"
        ]

        widgets = {

            "video": forms.FileInput(
                attrs={
                    "class": "form-control",
                    "accept": "video/*"
                }
            )

        }