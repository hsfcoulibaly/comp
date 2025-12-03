import math


def calculate_stereo_dimensions(
        fx: float,
        baseline_b: float,
        p1_l: tuple,
        p1_r: tuple,
        p2_edge1: tuple,
        p2_edge2: tuple,
        left_image_width: int
) -> dict:
    """
    Performs the two-step calculation for object distance (Z) and real-world size.

    This function assumes the input images are already rectified.

    Args:
        fx (float): Calibrated camera focal length in pixels.
        baseline_b (float): Distance between the two camera centers (in cm).
        p1_l (tuple): (x, y) pixel coordinates of the object's center in the LEFT image.
        p1_r (tuple): (x, y) pixel coordinates of the object's center in the RIGHT image.
        p2_edge1 (tuple): (x, y) pixel coordinates of the object's first edge (e.g., left side) in the LEFT image.
        p2_edge2 (tuple): (x, y) pixel coordinates of the object's second edge (e.g., right side) in the LEFT image.
        left_image_width (int): The original width of the left image in pixels (used for coordinate transformation).

    Returns:
        dict: Containing calculated Z, width, and any errors.
    """
    try:
        # --- STEP 1: Estimate Distance (Z) using Disparity ---

        # 1. Get X-coordinates for disparity
        x_L = p1_l[0]
        # X-coordinate in the Right Image's local coordinate system (must subtract Left image width)
        x_R = p1_r[0] - left_image_width

        # 2. Calculate Disparity (d = x_L - x_R)
        disparity_d = x_L - x_R

        if disparity_d <= 0:
            raise ValueError("Disparity is non-positive. Cannot estimate distance (Z).")

        #

        # 3. Apply the Stereo Distance Formula: Z = (f * B) / d
        calculated_z = (fx * baseline_b) / disparity_d

        # --- STEP 2: Estimate Object Size (Width) using Estimated Z ---

        # 1. Calculate Pixel Size (Width in pixels) from the Left Image
        pixel_width = abs(p2_edge2[0] - p2_edge1[0])

        # 2. Apply Monocular Size Formula: Real_Size = (Pixel_Size * Z) / f
        real_world_width = (pixel_width * calculated_z) / fx

        return {
            "status": "success",
            "z_distance": calculated_z,
            "real_world_width": real_world_width,
            "disparity": disparity_d,
            "unit": "cm"
        }

    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


if __name__ == '__main__':
    # --- Example Usage for Testing (Simulating Clicked Data) ---
    print("--- Testing Stereo Calculation Module ---")

    # Calibration Constants (Example values)
    EXAMPLE_FX = 1250.0  # Focal length in pixels
    EXAMPLE_BASELINE = 6.0  # Distance between cameras in cm
    LEFT_IMG_W = 640  # Width of the left image

    # Simulated Original Pixel Coordinates (Simulating 4 Clicks on a 640x480 Left image)
    # 1. P1_L (Center Left Image)
    center_l = (320, 240)
    # 2. P1_R (Center Right Image, relative to COMBINED image, which is 640+640=1280 wide)
    # The corresponding point in the right image is at x=320-15=305.
    # In the combined image, it's at 640 + 305 = 945
    center_r_combined = (945, 240)
    # 3. P2_Edge1 (Left edge of object in Left Image)
    edge_1 = (300, 240)
    # 4. P2_Edge2 (Right edge of object in Left Image)
    edge_2 = (340, 240)

    results = calculate_stereo_dimensions(
        fx=EXAMPLE_FX,
        baseline_b=EXAMPLE_BASELINE,
        p1_l=center_l,
        p1_r=center_r_combined,  # Note: Function handles conversion
        p2_edge1=edge_1,
        p2_edge2=edge_2,
        left_image_width=LEFT_IMG_W
    )

    if results['status'] == 'success':
        print(f"\nDisparity (d): {results['disparity']:.2f} pixels")
        print(f"Estimated Distance (Z): {results['z_distance']:.2f} {results['unit']}")
        print(f"Estimated Real-World Width: {results['real_world_width']:.2f} {results['unit']}")
    else:
        print(f"\nCalculation Failed: {results['message']}")