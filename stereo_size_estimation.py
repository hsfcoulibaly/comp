import cv2
import numpy as np

# --- 1. Global Variables and Callbacks ---
clicked_points = []
image_scale = 1.0  # Scale factor for display
left_image_width = 0 # Placeholder for left image width

def get_pixel_coordinates(event, x, y, flags, param):
    """ Captures and stores the pixel coordinates (x, y) when the left mouse button is clicked. """
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked on combined image (resized) at: x={x}, y={y}")

# --- 2. Main Execution and Display ---

# V V V V V V V V V V V V V V V V V V V V V V V V V V
# REPLACE with your image file names
image_path_L = "left.jpg"
image_path_R = "right.jpg"
# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

image_L = cv2.imread(image_path_L)
image_R = cv2.imread(image_path_R)

if image_L is None or image_R is None:
    print(f"Error: Could not load images. Please ensure '{image_path_L}' and '{image_path_R}' exist.")
else:
    # Ensure images have same height (essential for stereo)
    min_height = min(image_L.shape[0], image_R.shape[0])
    image_L = image_L[:min_height, :]
    image_R = image_R[:min_height, :]

    # Get original image dimensions (of the left image)
    orig_height, orig_width = image_L.shape[:2]
    left_image_width = orig_width
    print(f"Original Left Image dimensions: {orig_width}x{orig_height}")

    # Combine images for single-window interaction
    combined_image = np.hstack((image_L, image_R))

    # Simple scaling for display robustness
    MAX_DISPLAY_WIDTH = 1200
    if combined_image.shape[1] > MAX_DISPLAY_WIDTH:
        image_scale = MAX_DISPLAY_WIDTH / combined_image.shape[1]
        new_width = int(combined_image.shape[1] * image_scale)
        new_height = int(combined_image.shape[0] * image_scale)
        resized_image = cv2.resize(combined_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = combined_image
        image_scale = 1.0

    print(f"Combined image scaled by factor: {image_scale:.4f}")

    # --- Window Setup ---
    cv2.namedWindow("Stereo View - Click 4 Points", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Stereo View - Click 4 Points", get_pixel_coordinates)

    print("\n--- INSTRUCTIONS ---")
    print("1. Click the center point of the object in the LEFT image (P1_L for x_L).")
    print("2. Click the corresponding center point of the object in the RIGHT image (P1_R for x_R).")
    print("3. Click the LEFT edge of the object in the LEFT image (P2_edge1).")
    print("4. Click the RIGHT edge of the object in the LEFT image (P2_edge2).")
    print("Press any key after the 4 clicks to calculate.")

    cv2.imshow("Stereo View - Click 4 Points", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 3. Stereo Calculation Section ---
if len(clicked_points) >= 4:
    # V V V V V V V V V V V V V V V V V V V V V V V V V V
    # --- CALIBRATION PARAMETERS (UPDATE THESE) ---
    # CALIBRATED_FX: Focal length in pixels from stereo calibration (f_x)
    CALIBRATED_FX = 1250.0

    # BASELINE_B: Distance between the two camera lens centers (in cm, mm, etc.)
    BASELINE_B = 6.0
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

    # --- A. Un-scale all clicked points back to original pixel space ---
    inv_scale = 1.0 / image_scale
    p1_L_resized, p1_R_resized, p2_edge1_resized, p2_edge2_resized = clicked_points[:4]

    p1_L_original = (p1_L_resized[0] * inv_scale, p1_L_resized[1] * inv_scale)
    p1_R_original = (p1_R_resized[0] * inv_scale, p1_R_resized[1] * inv_scale)
    p2_edge1_original = (p2_edge1_resized[0] * inv_scale, p2_edge1_resized[1] * inv_scale)
    p2_edge2_original = (p2_edge2_resized[0] * inv_scale, p2_edge2_resized[1] * inv_scale)


    # --- B. STEP 1: Estimate Distance (Z) from Stereo Pair ---

    # 1. Get X-coordinates for disparity calculation
    x_L = p1_L_original[0]
    # x_R is relative to the right image's origin (must subtract left image width)
    x_R_combined = p1_R_original[0]
    x_R = x_R_combined - left_image_width

    # 2. Calculate Disparity (d = x_L - x_R)
    disparity_d = x_L - x_R

    if disparity_d <= 0:
        print("\n[ERROR] Disparity is non-positive. Check the clicking order (P1_L then P1_R) and if your points correspond.")
        CALCULATED_Z = 0.0
    else:
        # 3. Apply Stereo Distance Formula: Z = (f * B) / d
        CALCULATED_Z = (CALIBRATED_FX * BASELINE_B) / disparity_d

        print("\n--- STEREO DISTANCE (Z) ESTIMATION ---")
        print(f"X_L (Original): {x_L:.2f} px")
        print(f"X_R (Original, relative to R image): {x_R:.2f} px")
        print(f"Disparity (d): {disparity_d:.2f} px")
        print(f"Estimated Distance (Z): {CALCULATED_Z:.2f} (units match Baseline B)")


    # --- C. STEP 2: Estimate Object Size using Estimated Z ---
    if CALCULATED_Z > 0:
        # 1. Calculate Pixel Size (Width in pixels)
        pixel_difference_x = abs(p2_edge2_original[0] - p2_edge1_original[0])

        # 2. Apply Monocular Size Formula: Real_World_Size = (Pixel_Size * Z) / f
        # We use the Z calculated in the previous step.
        real_world_width = (pixel_difference_x * CALCULATED_Z) / CALIBRATED_FX

        print("\n--- OBJECT SIZE ESTIMATION ---")
        print(f"Pixel Size (Width): {pixel_difference_x:.2f} pixels")
        print(f"Using Estimated Z: {CALCULATED_Z:.2f} cm")
        print(f"\nCalculated Real-World Width: {real_world_width:.2f} cm (units match Z)")

else:
    print("\nCalculation skipped: You must click 4 points as instructed to perform the stereo calculation.")