import mediapipe as mp
import numpy as np
import cv2

from procrustes import Procrustes

hand_landmark_detector = mp.solutions.hands
hand_landmark_drawer = mp.solutions.drawing_utils
hand_drawing_styles = mp.solutions.drawing_styles

landmark_drawing_spec = hand_drawing_styles.get_default_hand_landmark_style()
connection_drawing_spec = hand_drawing_styles.get_default_hand_connection_style()

landmark_point_colors = np.array(
    [np.array(landmark_drawing_spec[point_idx].color) / 255.
     for point_idx in range(len(landmark_drawing_spec.keys()))])

landmark_connections_colors = np.array(
    [np.array(connection_drawing_spec[conn].color) / 255.
     for conn in hand_landmark_detector.HAND_CONNECTIONS])

PALM_LANDMARK_INDICES = [0, 1, 5, 9, 13, 17]
PALM_MODEL_POINTS = np.array(
    [[1.77144661e+02, 2.77453508e+02, -2.27418641e-03],
     [2.12762718e+02, 2.72557297e+02, -2.81604195e+01],
     [2.08728580e+02, 1.94294314e+02, -3.69296193e+01],
     [1.80297432e+02, 1.91437311e+02, -3.49721146e+01],
     [1.55069199e+02, 1.98956938e+02, -3.51533961e+01],
     [1.34010048e+02, 2.15013456e+02, -3.53264761e+01]], dtype=float)
PALM_MODEL_POINTS -= PALM_MODEL_POINTS.mean(axis=0)


class HandPose:
    def __init__(self):
        self.procrustes_transform = Procrustes()

    @staticmethod
    def detect_landmarks(frame):
        with hand_landmark_detector.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=.5,
                min_tracking_confidence=.5) as hands:
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks is None:
                return None

            return results

    @staticmethod
    def plot_landmarks_3d(landmarks, axis):
        axis.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], color=landmark_point_colors)
        connecting_lines = np.array([landmarks[list(conn)] for conn in hand_landmark_detector.HAND_CONNECTIONS])
        connecting_lines = np.rollaxis(connecting_lines, 2, 1)

        for line, color in zip(connecting_lines, landmark_connections_colors):
            x_, y_, z_ = line
            axis.plot(x_, y_, z_, color=color)

    @staticmethod
    def plot_landmarks_3d_plotly(landmarks, fig):
        points_trace = dict(type='scatter3d',
                            x=landmarks[:, 0],
                            y=landmarks[:, 1],
                            z=-landmarks[:, 2],
                            mode='markers',
                            marker=dict(
                                color=[f'rgb({c[0]}, {c[1]}, {c[2]})' for c in
                                       (landmark_point_colors * 255).astype(int)]))

        fig.add_trace(points_trace)

        connecting_lines = np.array([landmarks[list(conn)] for conn in hand_landmark_detector.HAND_CONNECTIONS])
        connecting_lines = np.rollaxis(connecting_lines, 2, 1)

        for line, color in zip(connecting_lines, landmark_connections_colors):
            x_, y_, z_ = line
            color = (color * 255).astype(int)
            connection_trace = dict(type='scatter3d',
                                    x=x_,
                                    y=y_,
                                    z=-z_,
                                    mode='lines',
                                    marker=dict(
                                        color=f'rgb({color[0]}, {color[1]}, {color[2]})'))
            fig.add_trace(connection_trace)
        fig.update_traces(line=dict(width=10))
        return fig

    @staticmethod
    def draw_detected_hand_landmarks(draw_layer, hand_landmark_detection):
        if not hand_landmark_detection:
            return

        for hand_landmarks in hand_landmark_detection.multi_hand_landmarks:
            hand_landmark_drawer.draw_landmarks(
                draw_layer, hand_landmarks, hand_landmark_detector.HAND_CONNECTIONS,
                landmark_drawing_spec,
                connection_drawing_spec)

    def normalize_hand_landmark_points(self, hand_landmarks, frame_shape):
        shape_y, shape_x = frame_shape[:2]
        landmark_scaling = np.array([shape_x, shape_y, shape_x])

        landmarks = np.array([[l.x, l.y, l.z] for l in hand_landmarks])
        landmarks *= landmark_scaling

        self.procrustes_transform.fit(PALM_MODEL_POINTS, landmarks[PALM_LANDMARK_INDICES])
        landmarks_transformed = self.procrustes_transform.transform(landmarks)

        return landmarks_transformed
