import cv2
import os
import mediapipe as mp

# Handle modern vs legacy MediaPipe solution imports
try:
    import mediapipe.python.solutions.face_detection as mp_face_detection
except ImportError:
    import mediapipe.solutions.face_detection as mp_face_detection

from django.conf import settings


def extract_faces_from_video(video_path, max_frames=10):
    """
    Extracts face crops from a video using MediaPipe Face Detection.
    """
    cap = cv2.VideoCapture(video_path)
    saved_face_paths = []

    output_dir = os.path.join(settings.MEDIA_ROOT, 'faces')
    os.makedirs(output_dir, exist_ok=True)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        frame_count = 0
        while cap.isOpened() and len(saved_face_paths) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 10th frame to avoid redundant crops
            if frame_count % 10 == 0:
                # Convert BGR (OpenCV) to RGB (MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    h, w, _ = frame.shape
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        xmin, ymin = int(bboxC.xmin * w), int(bboxC.ymin * h)
                        box_w, box_h = int(bboxC.width * w), int(bboxC.height * h)

                        # Keep coordinates within valid frame bounds
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(w, xmin + box_w), min(h, ymin + box_h)

                        face_crop = frame[ymin:ymax, xmin:xmax]
                        if face_crop.size > 0:
                            face_filename = f"face_{len(saved_face_paths) + 1}.jpg"
                            face_filepath = os.path.join(output_dir, face_filename)
                            cv2.imwrite(face_filepath, face_crop)
                            saved_face_paths.append(face_filepath)

                        if len(saved_face_paths) >= max_frames:
                            break
            frame_count += 1

    cap.release()
    return saved_face_paths