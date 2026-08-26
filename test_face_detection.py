import cv2

from detector.services.video_processor import extract_frames
from detector.services.face_detector import FaceDetector


video_path = "media/videos/test.mp4"


frames = extract_frames(
    video_path,
    num_frames=16
)


detector = FaceDetector()


faces_found = 0


for index, frame in enumerate(frames):

    face = detector.detect_face(frame)

    if face is not None:

        faces_found += 1

        print(
            f"Frame {index + 1}: Face detected"
        )

    else:

        print(
            f"Frame {index + 1}: No face"
        )


print()
print("Total frames:", len(frames))
print("Faces detected:", faces_found)