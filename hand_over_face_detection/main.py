import json
import os
import sys

import cv2
from tqdm import tqdm

sys.path.insert(1, '../AMD/head_pose_estimation')
from video_capture_eles import MyVideoCapture
from hand_over_face_detection_wrapper import HandOverFaceDetectionWrapper

# landin_path_template = 'C:\\Munka\\kim\\json_files\\{video_name}_landmarks.json'
# landin_path_template = 'C:\\Munka\\kim\\new_landmark_jsons\\{video_name}_landmarks_pfldst_dlib.json'
# landin_path_template = r'/home/revygabor/amd_project/landmarks/{video_name}_landmarks.json'

landin_path_template = r'/media/Data/Other/amd_project/landmarks/{video_name}_landmarks.json'
with open('video_params_hand_over_face.json', 'r') as openfile:
    video_params = json.load(openfile)['params']

video_params = video_params[30:]
for param in video_params:
    video_name = param["video_name"]
    print(video_name)
    landmark_file_path = landin_path_template.format(video_name=video_name)

    if not os.path.exists(landmark_file_path):
        print(f'landmark file for {video_name} doesn\'t exitst')
        continue

    vc = MyVideoCapture(video_path=param["vid_path"],
                        video_start_position=param["sfn"],
                        roi_left_upper_corner=param["fi"],
                        roi_right_lower_corner=param["ai"],
                        landmark_file_path=landmark_file_path,
                        video_end_position=param["ffn"])

    stream_open, frame = vc.read()
    landmarks = vc.readland()
    start_frame_index = int(vc.cap.get(cv2.CAP_PROP_FRAME_COUNT) * param["sfn"])
    end_frame_index = int(vc.cap.get(cv2.CAP_PROP_FRAME_COUNT) * param["ffn"])

    # Change this to run a different filter ##########################################################    
    current_filter = HandOverFaceDetectionWrapper(video_name)
    ##################################################################################################

    process_results = []
    pbar = tqdm(total=(end_frame_index - start_frame_index))
    while stream_open:
        pbar.update(1)
        filter_res = current_filter.process_frame(frame, landmarks)
        process_results.append(filter_res)

        stream_open, frame = vc.read()
        if stream_open:
            landmarks = vc.readland()

    pbar.close()
    vc.cap.release()
    current_filter.post_process(process_results)
