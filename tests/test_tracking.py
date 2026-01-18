from src.core.tracker import ByteTrack


def test_id_persistence():
    tracker = ByteTrack()
    detections = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0}]
    tracks = tracker.update(detections)
    assert len(tracks) == 1
    track_id = tracks[0].track_id

    tracks = tracker.update(detections)
    assert tracks[0].track_id == track_id


def test_trajectory_history():
    tracker = ByteTrack()
    detections = [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0}]
    tracker.update(detections)
    detections = [{"bbox": [5, 5, 15, 15], "confidence": 0.9, "class_id": 0}]
    tracks = tracker.update(detections)
    assert len(tracks[0].history) >= 2
