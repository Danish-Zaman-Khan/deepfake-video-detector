from django.shortcuts import render, redirect, get_object_or_404

from .forms import VideoUploadForm
from .models import VideoAnalysis


def home(request):

    if request.method == "POST":

        form = VideoUploadForm(
            request.POST,
            request.FILES
        )

        if form.is_valid():

            analysis = form.save()

            return redirect(
                "result",
                pk=analysis.pk
            )

    else:

        form = VideoUploadForm()


    return render(
        request,
        "detector/home.html",
        {
            "form": form
        }
    )


def result(request, pk):

    analysis = get_object_or_404(
        VideoAnalysis,
        pk=pk
    )

    return render(
        request,
        "detector/result.html",
        {
            "analysis": analysis
        }
    )