import numpy as np
import cv2
import matplotlib.pyplot as plt

from hand_pose import HandPose

fig = plt.figure('Left')
ax_left = fig.add_subplot()
ax_left = fig.add_subplot(projection='3d')
ax_left.view_init(elev=-90, azim=-90)


def main():
    cap = cv2.VideoCapture(0)
    hand_pose = HandPose()

    recorded_landmarks = []
    record_timer_counter = 1
    record_cycle = 3

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame = np.fliplr(frame)

        hand_landmark_detection = hand_pose.detect_landmarks(frame)

        frame_draw = frame.copy()
        draw_layer = np.zeros_like(frame_draw)

        if not hand_landmark_detection:
            continue

        landmarks_normalized = None
        for handedness_, hand in zip(hand_landmark_detection.multi_handedness,
                                     hand_landmark_detection.multi_hand_landmarks):
            handedness = handedness_.classification[0].label

            if handedness == 'Left':
                landmarks_normalized = hand_pose.normalize_hand_landmark_points(hand.landmark, frame_shape=frame.shape)
                ax_left.cla()
                hand_pose.plot_landmarks_3d(landmarks_normalized, ax_left)

                ax_left.set_xlim([-150, 150])
                ax_left.set_ylim([-200, 150])
                plt.pause(1e-9)

        if record_timer_counter == 0 and landmarks_normalized is not None:
            recorded_landmarks.append(landmarks_normalized.flatten())
            draw_layer[:10, :10, :0] = 255
        else:
            draw_layer = cv2.putText(draw_layer, str(record_cycle-record_timer_counter),
                                     (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 1, cv2.LINE_AA)

        # draw hand landmark detection on frame
        hand_pose.draw_detected_hand_landmarks(draw_layer, hand_landmark_detection)
        frame_draw = cv2.addWeighted(frame_draw, .6, draw_layer, .4, 0)
        cv2.imshow('frame', frame_draw)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

        record_timer_counter += 1
        record_timer_counter %= record_cycle

    cap.release()
    cv2.destroyAllWindows()

    np.save('normalized_landmarks.npy', np.array(recorded_landmarks), allow_pickle=False)


if __name__ == '__main__':
    main()
