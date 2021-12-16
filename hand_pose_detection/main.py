import numpy as np
import cv2
import matplotlib.pyplot as plt

from hand_pose import HandPose

fig = plt.figure('Left')
ax_left = fig.add_subplot()
ax_left = fig.add_subplot(projection='3d')
ax_left.view_init(elev=-90, azim=-90)


# def perp(a):
#     b = np.empty_like(a)
#     b[0] = -a[1]
#     b[1] = a[0]
#     return b
#
#
# def seg_intersect(a1, a2, b1, b2):
#     da = a2 - a1
#     db = b2 - b1
#     dp = a1 - b1
#     dap = perp(da)
#     denom = np.dot(dap, db)
#     num = np.dot(dap, dp)
#     return (num / denom.astype(float)) * db + b1


def main():
    cap = cv2.VideoCapture(0)
    hand_pose = HandPose()

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = np.fliplr(frame)

        hand_landmark_detection = hand_pose.detect_landmarks(frame)

        frame_draw = frame.copy()
        draw_layer = np.zeros_like(frame_draw)

        # draw hand landmark detection on frame
        hand_pose.draw_detected_hand_landmarks(draw_layer, hand_landmark_detection)
        frame_draw = cv2.addWeighted(frame_draw, .6, draw_layer, .4, 0)
        cv2.imshow('frame', frame_draw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('hand_landmarks.png', frame_draw)
            break

        if not hand_landmark_detection:
            continue
        for handedness_, hand in zip(hand_landmark_detection.multi_handedness,
                                     hand_landmark_detection.multi_hand_landmarks):
            handedness = handedness_.classification[0].label

            if handedness == 'Left':
                landmarks_transformed = hand_pose.normalize_hand_landmark_points(hand.landmark, frame_shape=frame.shape)
                ax_left.cla()
                hand_pose.plot_landmarks_3d(landmarks_transformed, ax_left)

                ax_left.set_xlim([-150, 150])
                ax_left.set_ylim([-200, 150])
                plt.pause(1e-9)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
