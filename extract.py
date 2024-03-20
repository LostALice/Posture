# Code by AkinoAlice@TyrantRey

from cv2.typing import MatLike

import mediapipe as mp
import numpy as np
import time
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

    def extract_pose(self, image: MatLike) -> tuple[MatLike, float | None]:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (500, 500))
        results = self.mp_pose.process(img)

        self.mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            self.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks is None:
            return img, None

        key_points = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])

        return img, key_points

    def calculate_similarity(self, pose_point: float, pose_point2: float) -> float:
        if pose_point is None or pose_point2 is None:
            # score 0
            return 0

        distances = np.linalg.norm(pose_point - pose_point2, axis=1)
        similarity = 1 / np.mean(distances)
        return similarity

if __name__ == "__main__":
    test_video = "./test.mp4"
    official_video = "./official.mp4"

    cap = cv2.VideoCapture(official_video)
    cap2 = cv2.VideoCapture(0)

    pose_detect = PostureDetection()
    while cap.isOpened() and cap2.isOpened():
        start = time.time()

        _, img = cap.read()
        _, img2 = cap2.read()

        if img is None and img2 is None:
            break

        processed_image, pose_point = pose_detect.extract_pose(img)
        processed_image2, pose_point2 = pose_detect.extract_pose(img2)
        print(processed_image, processed_image2)
        joined_image = cv2.vconcat([processed_image, processed_image2])

        fps = 1 / (time.time() - start)
        score = pose_detect.calculate_similarity(pose_point, pose_point2)

        cv2.putText(
            joined_image,
            f"FPS: {fps:.1f}| Score:{score:.1f}",
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.imshow("test", joined_image)

        if cv2.waitKey(5) == ord("q"):
            break
