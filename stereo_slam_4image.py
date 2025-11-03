import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

class StereoVO:
    def __init__(self, fx=531.695, fy=532.22, cx=627.105, cy=358.171,
                 k1=-0.0560871, k2=0.0299121, p1=-0.000203807, p2=0.000267721, k3=-0.0122448,
                 baseline=0.12):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.distCoeffs = np.array([k1, k2, p1, p2, k3])
        self.P1 = np.hstack((self.K, np.zeros((3, 1))))
        self.P2 = np.hstack((self.K, np.array([[-baseline * fx], [0], [0]])))
        self.baseline = baseline

        self.feature_extractor = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.pose = np.eye(4)
        self.trajectory = []

    def stereo_match(self, img_left, img_right, max_features=200):
        kpL, desL = self.feature_extractor.detectAndCompute(img_left, None)
        kpR, desR = self.feature_extractor.detectAndCompute(img_right, None)
        matches = self.matcher.match(desL, desR)
        matches = sorted(matches, key=lambda x: x.distance)[:max_features]
        return kpL, kpR, matches

    def temporal_match(self, img_left_prev, img_left_curr, max_features=200):
        kp_prev, des_prev = self.feature_extractor.detectAndCompute(img_left_prev, None)
        kp_curr, des_curr = self.feature_extractor.detectAndCompute(img_left_curr, None)
        matches = self.matcher.match(des_prev, des_curr)
        matches = sorted(matches, key=lambda x: x.distance)[:max_features]
        return kp_prev, kp_curr, matches

    def triangulate(self, kpL, kpR, matches):
        ptsL = np.float32([kpL[m.queryIdx].pt for m in matches]).T
        ptsR = np.float32([kpR[m.trainIdx].pt for m in matches]).T
        points4D = cv2.triangulatePoints(self.P1, self.P2, ptsL, ptsR)
        points3D = (points4D[:3] / points4D[3]).T
        return points3D

    def estimate_motion(self, points3D, kp_curr, matches_temp):
        pts2D = np.float32([kp_curr[m.trainIdx].pt for m in matches_temp])
        if len(points3D) > len(pts2D):
            points3D = points3D[:len(pts2D)]
        else:
            pts2D = pts2D[:len(points3D)]
        _, rvec, tvec, _ = cv2.solvePnPRansac(points3D, pts2D, self.K, None)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        self.pose = self.pose @ np.linalg.inv(T)
        self.trajectory.append(self.pose[:3, 3])
        return R, tvec

    def plot_trajectory(self):
        if len(self.trajectory) < 2:
            print("Not enough poses to plot.")
            return
        traj = np.array(self.trajectory)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Camera Trajectory")
        plt.show()


# ============================================================
# Load two stereo pairs
# ============================================================
left_images = sorted(glob.glob("selected_images/left*.png"))
right_images = sorted(glob.glob("selected_images/right*.png"))

imgL1 = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
imgR1 = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)
imgL2 = cv2.imread(left_images[5], cv2.IMREAD_GRAYSCALE)
imgR2 = cv2.imread(right_images[5], cv2.IMREAD_GRAYSCALE)

vo = StereoVO()

# Stereo matching (at time t)
kpL1, kpR1, stereo_matches = vo.stereo_match(imgL1, imgR1)

# Temporal matching (left_t → left_t+1)
kpL1_t, kpL2_t, temp_matches = vo.temporal_match(imgL1, imgL2)

# Triangulation + motion estimation
points3D = vo.triangulate(kpL1, kpR1, stereo_matches)
R, t = vo.estimate_motion(points3D, kpL2_t, temp_matches)

print("Rotation Matrix:\n", R)
print("Translation Vector:\n", t.T)

# ============================================================
# Create one large figure showing all 4 images + correspondences
# ============================================================

h, w = imgL1.shape
canvas = np.zeros((2*h, 2*w), dtype=np.uint8)

# Place images
canvas[0:h, 0:w] = imgL1          # top-left
canvas[0:h, w:2*w] = imgR1        # top-right
canvas[h:2*h, 0:w] = imgL2        # bottom-left
canvas[h:2*h, w:2*w] = imgR2      # bottom-right

canvas_color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

# Draw stereo matches (top row)
for m in stereo_matches[:80]:
    ptL = tuple(np.round(kpL1[m.queryIdx].pt).astype(int))
    ptR = tuple(np.round(kpR1[m.trainIdx].pt).astype(int) + np.array([w, 0]))
    cv2.line(canvas_color, ptL, ptR, (0, 255, 0), 1)
    cv2.circle(canvas_color, ptL, 2, (0, 255, 0), -1)
    cv2.circle(canvas_color, ptR, 2, (0, 255, 0), -1)

# Draw temporal matches (left column)
for m in temp_matches[:80]:
    pt_t = tuple(np.round(kpL1_t[m.queryIdx].pt).astype(int))
    pt_t1 = tuple(np.round(kpL2_t[m.trainIdx].pt).astype(int) + np.array([0, h]))
    cv2.line(canvas_color, pt_t, pt_t1, (255, 165, 0), 1)
    cv2.circle(canvas_color, pt_t, 2, (255, 165, 0), -1)
    cv2.circle(canvas_color, pt_t1, 2, (255, 165, 0), -1)

# Label quadrants
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(canvas_color, "Left_t", (50, 50), font, 1, (255, 255, 255), 2)
cv2.putText(canvas_color, "Right_t", (w+50, 50), font, 1, (255, 255, 255), 2)
cv2.putText(canvas_color, "Left_t+1", (50, h+50), font, 1, (255, 255, 255), 2)
cv2.putText(canvas_color, "Right_t+1", (w+50, h+50), font, 1, (255, 255, 255), 2)

# Show composite result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(canvas_color, cv2.COLOR_BGR2RGB))
plt.title("Stereo and Temporal Matching (t and t+1)")
plt.axis('off')
plt.show()

# Plot trajectory
vo.plot_trajectory()
