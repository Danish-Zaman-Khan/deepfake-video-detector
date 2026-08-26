from detector.services.video_processor import extract_frames


video_path = "media/videos/test.mp4"

frames = extract_frames(
    video_path,
    num_frames=16
)

print("Frames extracted:", len(frames))

for i, frame in enumerate(frames):

    print(
        f"Frame {i + 1}:",
        frame.shape
    )