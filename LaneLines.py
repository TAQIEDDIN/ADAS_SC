import cv2
import numpy as np
import matplotlib.image as mpimg


# ──────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────
LANE_FILL   = (0, 255, 0)
LANE_EDGE_L = (0, 255, 0)
LANE_EDGE_R = (0, 255, 0)
TEXT_PRIMARY   = (255, 255, 255)
TEXT_WARNING   = (239, 68, 68)    # red
TEXT_OK        = (34, 197, 94)    # green
ACCENT         = (34, 197, 94)    # green accent
HUD_BG         = (10, 60, 20)     # dark green tint
BORDER         = (0, 0, 220)      # red border (BGR)


def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img,  (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img,  (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img,  (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img,  (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90,  color, thickness)


def draw_filled_rounded_rect(img, pt1, pt2, color, radius=12, alpha=0.6):
    """Semi-transparent filled rounded rect via alpha blending."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_progress_bar(img, x, y, w, h, value, max_val, bar_color, bg_color=(40, 40, 60)):
    """Draw a horizontal progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    fill = int(w * min(abs(value) / max_val, 1.0))
    cv2.rectangle(img, (x, y), (x + fill, y + h), bar_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)


class LaneLines:
    def __init__(self):
        self.left_fit  = None
        self.right_fit = None
        self.binary    = None
        self.nonzero   = None
        self.nonzerox  = None
        self.nonzeroy  = None
        self.clear_visibility = True
        self.dir = []

        # ── Direction arrows ──────────────────────────────
        self.left_curve_img    = mpimg.imread("left_turn.png")
        self.right_curve_img   = mpimg.imread("right_turn.png")
        self.keep_straight_img = mpimg.imread("straight.png")

        for attr in ("left_curve_img", "right_curve_img", "keep_straight_img"):
            img = getattr(self, attr)
            setattr(self, attr,
                    cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))

        # ── Hyperparameters ───────────────────────────────
        self.nwindows = 9
        self.margin   = 100
        self.minpix   = 50

    # ─────────────────────────────────────────────────────
    def forward(self, img):
        self.extract_features(img)
        return self.fit_poly(img)

    # ─────────────────────────────────────────────────────
    def pixels_in_window(self, center, margin, height):
        topleft     = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)
        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        self.img           = img
        self.window_height = int(img.shape[0] // self.nwindows)
        self.nonzero  = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    # ─────────────────────────────────────────────────────
    def find_lane_pixels(self, img):
        out_img = np.dstack((img, img, img))

        histogram  = hist(img)
        midpoint   = histogram.shape[0] // 2
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current  = leftx_base
        rightx_current = rightx_base
        y_current      = img.shape[0] + self.window_height // 2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left  = (leftx_current,  y_current)
            center_right = (rightx_current, y_current)

            glx, gly = self.pixels_in_window(center_left,  self.margin, self.window_height)
            grx, gry = self.pixels_in_window(center_right, self.margin, self.window_height)

            leftx.extend(glx);  lefty.extend(gly)
            rightx.extend(grx); righty.extend(gry)

            if len(glx) > self.minpix:
                leftx_current  = int(np.mean(glx))
            if len(grx) > self.minpix:
                rightx_current = int(np.mean(grx))

        return leftx, lefty, rightx, righty, out_img

    # ─────────────────────────────────────────────────────
    def fit_poly(self, img):
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty)  > 1500:
            self.left_fit  = np.polyfit(lefty,  leftx,  2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        if self.left_fit is None or self.right_fit is None:
            return out_img

        maxy   = img.shape[0] - 1
        miny   = img.shape[0] // 3
        ploty  = np.linspace(miny, maxy, img.shape[0])

        left_fitx  = (self.left_fit[0]  * ploty**2 + self.left_fit[1]  * ploty + self.left_fit[2])
        right_fitx = (self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2])

        # ── Filled lane overlay (green) ───────────────────
        pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts       = np.hstack((pts_left, pts_right))
        overlay   = out_img.copy()
        cv2.fillPoly(overlay, np.int32(pts), LANE_FILL)
        cv2.addWeighted(overlay, 0.45, out_img, 0.55, 0, out_img)
        # ── Lane edge curves ──────────────────────────────
        for i in range(1, len(ploty)):
            y0, y1 = int(ploty[i-1]), int(ploty[i])

            # Left edge  – blue
            x0l, x1l = int(left_fitx[i-1]), int(left_fitx[i])
            cv2.line(out_img, (x0l, y0), (x1l, y1), LANE_EDGE_L, 6)

            # Right edge – green
            x0r, x1r = int(right_fitx[i-1]), int(right_fitx[i])
            cv2.line(out_img, (x0r, y0), (x1r, y1), LANE_EDGE_R, 6)

        return out_img

    # ─────────────────────────────────────────────────────
    def plot(self, out_img):
        # ── No detection fallback ─────────────────────────
        if self.left_fit is None or self.right_fit is None:
            draw_filled_rounded_rect(out_img, (10, 10), (420, 70), HUD_BG, alpha=0.8)
            draw_rounded_rect(out_img, (10, 10), (420, 70), TEXT_WARNING, 2)
            cv2.putText(out_img, "⚠  Lane Not Detected", (20, 52),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, TEXT_WARNING, 2)
            return out_img

        lR, rR, pos = self.measure_curvature()

        # ── Direction detection ───────────────────────────
        value = (self.left_fit[0]
                 if abs(self.left_fit[0]) > abs(self.right_fit[0])
                 else self.right_fit[0])

        self.dir.append("F" if abs(value) <= 0.00015 else ("L" if value < 0 else "R"))
        if len(self.dir) > 10:
            self.dir.pop(0)
        direction = max(set(self.dir), key=self.dir.count)

        # ════════════════════════════════════════════════
        #  HUD PANEL  — compact, rounded, green+red
        # ════════════════════════════════════════════════
        PAD   = 12          # margin from top-left corner
        W, H  = 300, 390    # smaller panel
        R     = 28          # big corner radius → rounder look
        x1, y1 = PAD, PAD
        x2, y2 = PAD + W, PAD + H

        # Semi-transparent dark-green background
        draw_filled_rounded_rect(out_img, (x1, y1), (x2, y2), HUD_BG, radius=R, alpha=0.72)
        # Red border  (2 px)
        draw_rounded_rect(out_img, (x1, y1), (x2, y2), BORDER, 2, radius=R)

        # ── Header strip ──────────────────────────────────
        cv2.putText(out_img, "LANE ASSIST", (x1 + 16, y1 + 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, TEXT_PRIMARY, 1)

        # ── Direction arrow image ─────────────────────────
        if direction == "L":
            arrow_img = self.left_curve_img
            dir_label = "Left Curve Ahead"
            dir_color = (255, 180, 50)          # warm yellow (BGR)
        elif direction == "R":
            arrow_img = self.right_curve_img
            dir_label = "Right Curve Ahead"
            dir_color = (255, 180, 50)
        else:
            arrow_img = self.keep_straight_img
            dir_label = "Keep Straight"
            dir_color = TEXT_OK

        # Scale arrow to fit inside panel (max 110×110) then blit
        ah, aw = arrow_img.shape[:2]
        scale  = min(110 / aw, 110 / ah)
        new_w, new_h = int(aw * scale), int(ah * scale)
        small_arrow  = cv2.resize(arrow_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ax = x1 + (W - new_w) // 2      # centered horizontally
        ay = y1 + 54                     # just below header

        alpha_ch = small_arrow[:, :, 3] / 255.0
        for c in range(3):
            out_img[ay:ay+new_h, ax:ax+new_w, c] = (
                alpha_ch * small_arrow[:, :, c] +
                (1 - alpha_ch) * out_img[ay:ay+new_h, ax:ax+new_w, c]
            ).astype(np.uint8)

        # Direction label
        (tw, _), _ = cv2.getTextSize(dir_label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.putText(out_img, dir_label, (x1 + (W - tw) // 2, ay + new_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, dir_color, 2)

        # ── Divider ───────────────────────────────────────
        div_y = ay + new_h + 36
        cv2.line(out_img, (x1 + 16, div_y), (x2 - 16, div_y), BORDER, 1)

        # ── Curvature ─────────────────────────────────────
        cv2.putText(out_img, "Curvature Radius", (x1 + 14, div_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, ACCENT, 1)
        cv2.putText(out_img, f"L:{lR:>7.0f}m  R:{rR:>7.0f}m", (x1 + 14, div_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, TEXT_PRIMARY, 1)

        # ── Offset bar ────────────────────────────────────
        cv2.putText(out_img, f"Offset: {pos:+.2f} m", (x1 + 14, div_y + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_PRIMARY, 1)

        bar_color = TEXT_WARNING if abs(pos) > 0.8 else TEXT_OK
        draw_progress_bar(out_img, x1 + 14, div_y + 76, W - 30, 10, pos, 1.5, bar_color)

        # ── Lane-departure status ─────────────────────────
        cv2.line(out_img, (x1 + 16, div_y + 100), (x2 - 16, div_y + 100), BORDER, 1)

        if abs(pos) > 0.8:
            status_text  = "LANE DEPARTURE!"
            status_color = TEXT_WARNING
            draw_rounded_rect(out_img, (x1, y1), (x2, y2), TEXT_WARNING, 3, radius=R)
        else:
            status_text  = "Good Lane Keeping"
            status_color = TEXT_OK

        (sw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.putText(out_img, status_text, (x1 + (W - sw) // 2, div_y + 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

        return out_img

    # ─────────────────────────────────────────────────────
    def measure_curvature(self):
        ym = 30 / 720
        xm = 3.7 / 700

        y_eval = 700 * ym

        left_curveR  = ((1 + (2 * self.left_fit[0]  * y_eval + self.left_fit[1])  ** 2) ** 1.5) \
                       / np.absolute(2 * self.left_fit[0])
        right_curveR = ((1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** 1.5) \
                       / np.absolute(2 * self.right_fit[0])

        xl  = np.dot(self.left_fit,  [700 ** 2, 700, 1])
        xr  = np.dot(self.right_fit, [700 ** 2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm

        return left_curveR, right_curveR, pos