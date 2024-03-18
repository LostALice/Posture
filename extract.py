# Code by AkinoAlice@TyrantRey

from cv2.typing import MatLike

import mediapipe as mp
import numpy as np
import cv2


class PostureDetection(object):
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.min_detection_confidence = 0.3
        self.min_tracking_confidence = 0.3

        self.pose = mp.solutions.pose
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)

    def extract_pose(self, image: MatLike) -> MatLike:
        # convert image to rgb and resize
        img = cv2.resize(image, (500, 500))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(img)

        self.mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            self.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        return img


if __name__ == "__main__":
    test_video = "./test.mp4"
    official_video = "./official.mp4"

    cap = cv2.VideoCapture(official_video)
    cap2 = cv2.VideoCapture(test_video)

    pose_detect = PostureDetection()
    while cap.isOpened() and cap2.isOpened():
        _, img = cap.read()
        __, img2 = cap2.read()
        # if _ and __:
        processed_image = pose_detect.extract_pose(img)
        processed_image2 = pose_detect.extract_pose(img2)
        joined_image = np.concatenate((processed_image, processed_image2), axis=1)
        cv2.imshow("test", joined_image)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止