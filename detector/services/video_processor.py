import os
from .predictor import predict_face

def process_video_faces(face_image_paths):
    """
    Passes extracted face crops through the predictor and computes an overall score.
    """
    if not face_image_paths:
        return {'is_fake': False, 'average_confidence': 0.0, 'processed_faces': 0}

    results = []
    for face_path in face_image_paths:
        res = predict_face(face_path)
        results.append(res['confidence'])

    # Calculate overall confidence across all face crops
    avg_confidence = sum(results) / len(results)
    
    return {
        'is_fake': avg_confidence > 0.5,
        'average_confidence': round(avg_confidence, 4),
        'processed_faces': len(results)
    }