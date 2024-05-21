# Code by AkinoAlice@TyrantRey

from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.graphics.shapes import Drawing

from cv2.typing import MatLike
from pytube import YouTube
from typing import Tuple

import mediapipe as mp
import numpy as np
import time
import cv2

class VideoDownloader(object):
    def __init__(self, youtube_video: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ") -> None:
        self.video_url = youtube_video

    def download(self) -> str:
        yt = YouTube(self.video_url).streams.filter(file_extension="mp4")
        yt.first().download("./{yt.default_filename}.mp4")
        return yt.default_filename

class Report(object):
    def __init__(self, source_video: MatLike, passing_score: float = 50.) -> None:
        self.report_name = time.time()
        self.length = int(source_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = source_video.get(cv2.CAP_PROP_FPS)
        self.passing_score = passing_score

        self.score_time_line = []
        self.fail_count = 0

    def add_frame(self, score: float = 0.) -> None:
        self.score_time_line.append(score)

        if score < self.passing_score:
            self.fail_count += 1

    def generate_report(self, path: str = "./report.pdf") -> str:
        pdf = SimpleDocTemplate(path)
        content = []
        content.append(Paragraph("Dancing Score"))

        # graph
        drawing = Drawing(200, 100)
        data = [self.score_time_line]

        horizontalLineChart = HorizontalLineChart()
        horizontalLineChart.data = data

        drawing.add(horizontalLineChart)
        content.append(drawing)

        # statistics
        content.append(Paragraph(f"Average Score: {sum(self.score_time_line)/len(self.score_time_line):.2f}"))
        content.append(Paragraph(f"Average passing rate: {self.fail_count/len(self.score_time_line)}%"))

        pdf.build(content)

        return path

class PostureDetection(object):
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.image_size = (600, 600)
        self.min_detection_confidence = 0.3
        self.min_tracking_confidence = 0.3

        self.pose = mp.solutions.pose
        self.mp_pose = mp.solutions.pose.Pose(
            enable_segmentation=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)

        #     cv2.imread("./background.jpg"), (self.image_size[1], self.image_size[0]))
        self.background = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

    def extract_pose(self, image: MatLike, remove_bg: bool = True) -> Tuple[MatLike, float]:
        img = cv2.resize(image, self.image_size)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.mp_pose.process(rgb_img)

        if not results.pose_landmarks is None and remove_bg:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            img = np.where(condition, img, self.background)

        if results.pose_landmarks is None:
            return img, None

        self.mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            self.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

        key_points = np.array([[landmark.x, landmark.y]
                              for landmark in results.pose_landmarks.landmark])

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
    cap2 = cv2.VideoCapture(test_video)

    pose_detect = PostureDetection()
    reporter = Report(cap, 50)


    try:
        while cap.isOpened() and cap2.isOpened():
            # skip frames
            start = time.time()

            _, img = cap.read()
            _, img2 = cap2.read()

            if img is None and img2 is None:
                break

            processed_image, pose_point = pose_detect.extract_pose(img)
            processed_image2, pose_point2 = pose_detect.extract_pose(img2)

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

            reporter.add_frame(score)

            if cv2.waitKey(5) == ord("q"):
                break
    except:
        pass
    finally:
        pdf_path = reporter.generate_report()
        print(f"report generated at {pdf_path}")
