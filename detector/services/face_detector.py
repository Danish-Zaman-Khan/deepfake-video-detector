import torch

from facenet_pytorch import MTCNN


class FaceDetector:

    def __init__(self):

        # Use CPU for MTCNN because of MPS
        # adaptive pooling compatibility issues.
        self.device = torch.device("cpu")

        self.detector = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            device=self.device
        )

    def detect_face(self, frame):

        # OpenCV uses BGR.
        # MTCNN expects RGB.
        rgb_frame = frame[:, :, ::-1]

        face = self.detector(
            rgb_frame
        )

        return face