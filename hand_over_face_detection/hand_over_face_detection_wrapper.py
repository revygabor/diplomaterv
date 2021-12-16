import json
import pathlib
from typing import Any, List
import sys

import cv2
import numpy as np

sys.path.insert(1, '../AMD/head_pose_estimation')
from filter_base import FilterBase
from modules.hand_over_face_detector import HandOverFaceDetector

movement_length = 25


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class HandOverFaceDetectionWrapper(FilterBase):
    def __init__(self, video_name: str):
        self.video_name = video_name

        # preprocess
        self.hand_over_face_detector = HandOverFaceDetector()

        # post process
        #

    def process_frame(self, frame: np.ndarray, landmarks: dict) -> Any:
        landmarks = landmarks['pfld']
        if landmarks is None:
            return None

        landmarks = np.array(landmarks)

        res = {}

        hand_landmark_detection = self.hand_over_face_detector.detect_hand_landmarks(frame)
        hand_over_face_detection = self.hand_over_face_detector.detect_hand_over_face(frame, landmarks, hand_landmark_detection)
        self.hand_over_face_detector.draw_detection(frame, landmarks, hand_landmark_detection, hand_over_face_detection)

        return res

    def post_process(self, results: List) -> None:
        return