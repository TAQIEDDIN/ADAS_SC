"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH
    main.py --live INPUT_PATH

Options:
    -h --help     show this screen
    --video       process video file and save output
    --live        process video file and show live window
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from moviepy.editor import VideoFileClip

from CameraCalibration import CameraCalibration
from Thresholding import Thresholding
from PerspectiveTransformation import PerspectiveTransformation
from LaneLines import LaneLines


class FindLaneLines:
    def __init__(self):
        self.calibration = CameraCalibration("camera_cal", 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)

        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.shape[:2] != out_img.shape[:2]:
            img = cv2.resize(img, (out_img.shape[1], out_img.shape[0]))

        out_img = cv2.addWeighted(out_img, 1.0, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img

    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.forward)
        out_clip.write_videofile(output_path, audio=False)

    def process_video_live(self, input_path):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: cannot open video {input_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV كيقرى BGR، نحولوه RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed = self.forward(frame_rgb)

            # رجع BGR باش cv2.imshow يعرضو مزيان
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            cv2.imshow("Lane Detection Live", processed_bgr)

            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    args = docopt(__doc__)
    find_lane_lines = FindLaneLines()

    if args["--live"]:
        input_path = args["INPUT_PATH"]
        find_lane_lines.process_video_live(input_path)
    else:
        input_path = args["INPUT_PATH"]
        output_path = args["OUTPUT_PATH"]

        if args["--video"]:
            find_lane_lines.process_video(input_path, output_path)
        else:
            find_lane_lines.process_image(input_path, output_path)


if __name__ == "__main__":
    main()