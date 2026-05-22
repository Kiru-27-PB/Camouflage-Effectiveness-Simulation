import cv2
import numpy as np

# =========================
# LOAD BACKGROUND IMAGE
# =========================
background = cv2.imread("C:\\# My_Projects\\College Projects\\Camouflage Effectiveness Simulation (P)\\Simulation_Core\\Images\\B1.png")

# Check if image loaded
if background is None:
    print("Error loading background image")
    exit()

# =========================
# LOAD OBJECT IMAGE (PNG)
# =========================
object_img = cv2.imread("C:\\# My_Projects\\College Projects\\Camouflage Effectiveness Simulation (P)\\Simulation_Core\\Images\\S1.png", cv2.IMREAD_UNCHANGED)

# Check if image loaded
if object_img is None:
    print("Error loading object image")
    exit()

# =========================
# RESIZE OBJECT
# =========================
scale_percent = 30

new_width = int(object_img.shape[1] * scale_percent / 100)
new_height = int(object_img.shape[0] * scale_percent / 100)

object_img = cv2.resize(object_img, (new_width, new_height))

# =========================
# SPLIT CHANNELS
# =========================
# PNG contains:
# B G R A
# A = Alpha (transparency)

b, g, r, a = cv2.split(object_img)

# Merge BGR channels
object_rgb = cv2.merge((b, g, r))

# Create mask from alpha channel
mask = cv2.merge((a, a, a))

# Normalize mask
mask = mask / 255.0

# =========================
# CHOOSE POSITION
# =========================
x = 300
y = 600

# Object dimensions
h, w = object_rgb.shape[:2]

# =========================
# REGION OF INTEREST (ROI)
# =========================
roi = background[y:y+h, x:x+w]

# =========================
# BLENDING
# =========================
blended = roi * (1 - mask) + object_rgb * mask

# Convert back to uint8
blended = blended.astype(np.uint8)

# Put blended image back
background[y:y+h, x:x+w] = blended

# =========================
# EDGE DETECTION
# =========================

# Convert image to grayscale
gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 100, 200)

# Show edges
cv2.imshow("Edge Detection", edges)

# Count edge pixels
edge_pixels = np.sum(edges > 0)

total_pixels = edges.shape[0] * edges.shape[1]

edge_density = edge_pixels / total_pixels

print("Edge Density:", edge_density)

print("Edge Pixels:", edge_pixels)

# =========================
# COLOR DIFFERENCE
# =========================

# Mean color of ROI
roi_mean = np.mean(roi)

# Mean color of object
object_mean = np.mean(object_rgb)

# Difference
color_difference = abs(roi_mean - object_mean)

print("Color Difference:", color_difference)

# =========================
# BRIGHTNESS DIFFERENCE
# =========================

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
object_gray = cv2.cvtColor(object_rgb, cv2.COLOR_BGR2GRAY)

brightness_difference = abs(
    np.mean(roi_gray) - np.mean(object_gray)
)

print("Brightness Difference:", brightness_difference)

# =========================
# CAMOUFLAGE SCORE
# =========================

score = 100

score -= edge_density * 50
score -= color_difference * 0.5
score -= brightness_difference * 0.5

# Clamp score
score = max(0, min(100, score))

print(f"Camouflage Effectiveness: {score:.2f}%")

# =========================
# SHOW RESULT
# =========================
cv2.imshow("Camouflage Simulation", background)

# =========================
# SAVE OUTPUT
# =========================
cv2.imwrite("C:\\# My_Projects\\College Projects\\running [ hold ]\\Simulation_Core\\Output\\final_output.jpg", background)

print("Output saved in Output folder")

# Wait until key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()