import numpy as np
import cv2
import glob
import matplotlib.image as mpimg


class CameraCalibration:
    """Calibrate camera using chessboard images."""

    def __init__(self, image_dir, nx, ny, debug=False):
        """
        Parameters:
            image_dir (str): path to folder containing chessboard images
            nx (int): number of inner corners along width
            ny (int): number of inner corners along height
            debug (bool): unused حاليا، خليتها باش يبقى compatible
        """
        fnames = glob.glob(f"{image_dir}/*")

        if not fnames:
            raise Exception(f"No calibration images found in folder: {image_dir}")

        objpoints = []
        imgpoints = []

        # 3D points in real world space
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        img_shape = None

        for f in fnames:
            img = mpimg.imread(f)

            # بعض الصور كيجيو float [0,1] من mpimg، نحولوهم لـ uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # إذا كانت الصورة RGBA نحولها RGB
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # مهم: corners خاصها gray ماشي img
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
                img_shape = (img.shape[1], img.shape[0])

        if len(objpoints) == 0 or len(imgpoints) == 0:
            raise Exception("Unable to detect chessboard corners in calibration images.")

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )

        if not ret:
            raise Exception("Unable to calibrate camera.")

    def undistort(self, img):
        """
        Return undistorted image.

        Parameters:
            img (np.array): input image

        Returns:
            np.array: undistorted image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)