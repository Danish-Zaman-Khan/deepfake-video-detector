import time
from django.shortcuts import render, redirect, get_object_or_404
from .forms import VideoUploadForm
from .models import VideoAnalysis
from .services.face_detector import extract_faces_from_video
from .services.predictor import predict_face


def home(request):
    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            start_time = time.time()

            # 1. Save upload instance to DB
            analysis = form.save()
            video_path = analysis.video.path

            # 2. Extract face crops from video
            face_paths = extract_faces_from_video(video_path)

            # 3. Run predictions on extracted face crops
            if face_paths:
                confidences = []
                for face_path in face_paths:
                    result = predict_face(face_path)
                    confidences.append(result['confidence'])

                avg_confidence = sum(confidences) / len(confidences)
                is_fake = avg_confidence > 0.5

                analysis.prediction = "FAKE" if is_fake else "REAL"
                analysis.confidence = round(avg_confidence, 4)
                analysis.frames_analyzed = len(face_paths)
            else:
                # If MediaPipe/OpenCV finds no face, mark explicitly
                analysis.prediction = "NO_FACE_DETECTED"
                analysis.confidence = 0.0
                analysis.frames_analyzed = 0

            analysis.processing_time = round(time.time() - start_time, 2)
            analysis.model_version = "ResNet50_v1"
            analysis.save()  # <--- CRITICAL: Ensures DB is updated before redirect

            return redirect("result", pk=analysis.pk)
    else:
        form = VideoUploadForm()

    return render(
        request,
        "detector/home.html",
        {"form": form}
    )


def result(request, pk):
    analysis = get_object_or_404(VideoAnalysis, pk=pk)

    return render(
        request,
        "detector/result.html",
        {"analysis": analysis}
    )