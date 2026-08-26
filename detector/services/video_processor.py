import cv2


def extract_frames(video_path, num_frames=16):

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video.")

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames == 0:
        capture.release()
        raise ValueError("Video contains no frames.")

    frame_indices = [
        int(i * total_frames / num_frames)
        for i in range(num_frames)
    ]

    frames = []

    for index in frame_indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            index
        )

        success, frame = capture.read()

        if success:
            frames.append(frame)

    capture.release()

    return frames