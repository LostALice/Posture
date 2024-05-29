# Code by AkinoAlice@TyrantRey

from flask import Flask, request, render_template, send_file, redirect, url_for
from posture import *


app = Flask(__name__)

def process_video():
    test_video = "./video/uploaded_video.mp4"
    # test_video = "./video/youtube.mp4"
    official_video = "./video/youtube.mp4"

    cap = cv2.VideoCapture(official_video)
    cap2 = cv2.VideoCapture(test_video)

    pose_detect = PostureDetection()
    reporter = Report(cap, 50)

    try:
        count = 0
        while cap.isOpened() and cap2.isOpened():
            count += 1
            start = time.time()

            _, img = cap.read()
            _, img2 = cap2.read()

            if img is None and img2 is None:
                break

            processed_image, pose_point = pose_detect.extract_pose(img)
            processed_image2, pose_point2 = pose_detect.extract_pose(img2)

            # joined_image = cv2.vconcat([processed_image, processed_image2])

            fps = 1 / (time.time() - start)
            score = pose_detect.calculate_similarity(pose_point, pose_point2)
            print(count, fps, score)

            # cv2.putText(
            #     joined_image,
            #     f"FPS: {fps:.1f}| Score:{score:.1f}",
            #     (50, 50),
            #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #     1,
            #     (255, 255, 255),
            #     1,
            #     cv2.LINE_AA
            # )
            # cv2.imshow("test", joined_image)

            reporter.add_frame(score)

            # if cv2.waitKey(5) == ord("q"):
            #     break
    except:
        pass
    finally:
        pdf_path = reporter.generate_report()
        print(f"report generated at {pdf_path}")
        return pdf_path

@app.route("/")
async def index():
    return render_template("index.html")

@app.route("/results")
async def results():
    return send_file("./report.pdf", as_attachment=True)

@app.route("/upload", methods=["POST"])
async def upload_video():
    if "video" not in request.files:
        return "No video file in request", 400
    if "url" not in request.form:
        return "No video URL in request", 400

    url = request.form["url"]
    video = request.files["video"]
    VideoDownloader(url).download()
    video.save("./video/uploaded_video.mp4")

    process_video()

    return "success", 200


if __name__ == "__main__":
    app.run(debug=True)
