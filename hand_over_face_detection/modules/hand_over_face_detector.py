import mediapipe as mp
import numpy as np
import cv2


class HandOverFaceDetector:
    def __init__(self):
        self.hand_landmark_detector = mp.solutions.hands
        self.hand_landmark_drawer = mp.solutions.drawing_utils
        self.hand_drawing_styles = mp.solutions.drawing_styles
        self.face_regions = {
            "face_top": [86, 17, 72, 1, 49, 104],  # landmarks to compute region
            "face_middle": [14, 1, 72, 17, 30],
            "face_bottom": [0, 6, 2, 14, 30, 18, 22]
        }

    def detect_hand_landmarks(self, frame):
        with self.hand_landmark_detector.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=.5,
                min_tracking_confidence=.5) as hands:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks is None:
            return None

        return results

    def detect_hand_over_face(self, frame, face_landmarks, hand_landmark_detection):
        detection = {}

        # compute face regions
        region_landmarks = {}
        for region in self.face_regions.keys():
            region_landmarks[f'{region}'] = np.array([face_landmarks[l] for l in self.face_regions[region]])
        region_landmarks['face_top'][-2:] -= region_landmarks['face_top'][0] - region_landmarks['face_top'][2]
        region_landmarks['face_top'] = region_landmarks['face_top'][1:]

        for k in region_landmarks.keys():
            detection[f'{k}_region'] = region_landmarks[k]

        # extract landmark points of the hands
        hand_landmarks = None
        if hand_landmark_detection is not None:
            hand_landmarks = []
            for hand in hand_landmark_detection.multi_hand_landmarks:
                for l in hand.landmark:
                    hand_landmarks.append([l.x, l.y])
            hand_landmarks = np.array(hand_landmarks)
            height, width, _ = frame.shape
            hand_landmarks *= np.array([width, height])
            hand_landmarks = hand_landmarks.round(decimals=0).astype(int)

        for region in self.face_regions.keys():
            detection[f'{region}_occluded'] = False
            if hand_landmarks is not None:
                region_polygon = region_landmarks[f'{region}']
                for hand_landmark_point in hand_landmarks[::2]:
                    if cv2.pointPolygonTest(region_polygon, hand_landmark_point.astype(float), measureDist=False) >= 0:
                        detection[f'{region}_occluded'] = True
                        break

        return detection

    def draw_detection(self, frame, face_landmarks, hand_landmark_detection, hand_over_face_detection):
        frame_draw = frame.copy()
        draw_layer = np.zeros_like(frame_draw)

        for region in self.face_regions.keys():
            if hand_over_face_detection[f'{region}_occluded']:
                color = (128, 128, 256)
            else:
                color = (128, 256, 0)
            cv2.fillPoly(draw_layer, pts=[hand_over_face_detection[f'{region}_region']], color=color)
        if hand_landmark_detection:
            for hand_landmarks in hand_landmark_detection.multi_hand_landmarks:
                self.hand_landmark_drawer.draw_landmarks(
                    draw_layer, hand_landmarks, self.hand_landmark_detector.HAND_CONNECTIONS,
                    self.hand_drawing_styles.get_default_hand_landmark_style(),
                    self.hand_drawing_styles.get_default_hand_connection_style())

        layer_mask = ~np.all(draw_layer == np.array([0, 0, 0]), axis=-1)
        # frame_draw = cv2.addWeighted(frame_draw, .6, draw_layer, .4, 0)
        frame_draw[layer_mask] = cv2.addWeighted(frame_draw, .6, draw_layer, .4, 0)[layer_mask]

        cv2.imshow('face occlusion detection', frame_draw)
        cv2.waitKey(40)
        print('', end='')
