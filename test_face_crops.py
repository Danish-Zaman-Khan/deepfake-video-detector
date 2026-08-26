import cv2
import os

from detector.services.video_processor import extract_frames
from detector.services.face_detector import FaceDetector


VIDEO_PATH = "media/videos/test.mp4"
OUTPUT_DIR = "media/faces"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


frames = extract_frames(
    VIDEO_PATH,
    num_frames=16
)


detector = FaceDetector()


faces_found = 0


for index, frame in enumerate(frames):

    face = detector.detect_face(frame)

    if face is None:
        print(f"Frame {index + 1}: No face")
        continue

    # MTCNN returns a PyTorch tensor.
    # Convert it to a NumPy image.
    face = face.permute(1, 2, 0)

    face = face.detach().cpu().numpy()

    # Convert normalized values to 0–255
    face = ((face + 1) / 2 * 255)

    face = face.clip(0, 255).astype("uint8")

    # RGB → BGR for OpenCV
    face = cv2.cvtColor(
        face,
        cv2.COLOR_RGB2BGR
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"face_{index + 1}.jpg"
    )

    cv2.imwrite(
        output_path,
        face
    )

    faces_found += 1

    print(
        f"Frame {index + 1}: "
        f"Face saved → {output_path}"
    )


print()
print("Total frames:", len(frames))
print("Faces saved:", faces_found)