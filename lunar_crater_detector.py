import cv2
import numpy as np
import os

# Default settings
DEFAULT_IMAGE_PATH = "LunarLandingSiteNavigator3/graphics/maps/global_map.png"
MOON_RADIUS_KM = 1737.4
DEFAULT_PATCH_WIDTH_KM = 1500
DEFAULT_OUTPUT_SIZE = 1024


def create_local_projection(
    image,
    centre_lat,
    centre_lon,
    patch_width_km=DEFAULT_PATCH_WIDTH_KM,
    output_size=DEFAULT_OUTPUT_SIZE
):
    height, width = image.shape

    lat0 = np.radians(centre_lat)
    lon0 = np.radians(centre_lon)

    # Coordinates in kilometres
    km_per_pixel = patch_width_km / output_size

    x = (
        np.arange(output_size) - (output_size - 1) / 2
    ) * km_per_pixel

    y = (
        (output_size - 1) / 2 - np.arange(output_size)
    ) * km_per_pixel

    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X**2 + Y**2) / MOON_RADIUS_KM
    c = rho
    safe_rho = np.where(rho == 0, 1, rho)

    lat = np.arcsin(
        np.cos(c) * np.sin(lat0)
        + (Y / MOON_RADIUS_KM)
        * np.sin(c) * np.cos(lat0) / safe_rho
    )

    lon = lon0 + np.arctan2(
        X * np.sin(c),
        MOON_RADIUS_KM * safe_rho * np.cos(lat0) * np.cos(c)
        - Y * np.sin(lat0) * np.sin(c)
    )

    lat[rho == 0] = lat0
    lon[rho == 0] = lon0

    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    # Map geographic coordinates to source pixels
    map_x = ((lon_deg + 180) / 360) * (width - 1)
    map_y = ((90 - lat_deg) / 180) * (height - 1)

    # Wrap longitude across the map seam
    map_x = np.mod(map_x, width)

    # Sample the original image
    projected = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return projected


def generate_basin_detection(
    centre_lat,
    centre_lon,
    image_path=DEFAULT_IMAGE_PATH,
    projection_output_path="LunarLandingSiteNavigator3/lunar_results/corrected_projection.png",
    detection_output_path="LunarLandingSiteNavigator3/graphics/impact_basin_detection.png",
    patch_width_km=DEFAULT_PATCH_WIDTH_KM,
    output_size=DEFAULT_OUTPUT_SIZE,
    verbose=False
):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot find {image_path}. Check the file location."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    projected = create_local_projection(
        gray,
        centre_lat,
        centre_lon,
        patch_width_km,
        output_size
    )

    if projection_output_path:
        os.makedirs(os.path.dirname(projection_output_path) or ".", exist_ok=True)
        cv2.imwrite(projection_output_path, projected)

    blurred = cv2.GaussianBlur(projected, (11, 11), 3)

    DARK_PERCENTILE = 30
    threshold = np.percentile(blurred, DARK_PERCENTILE)

    _, binary = cv2.threshold(
        blurred,
        int(threshold),
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Remove small isolated regions
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = cv2.cvtColor(projected, cv2.COLOR_GRAY2BGR)
    detections = []
    MIN_AREA = 1000
    km_per_pixel = patch_width_km / output_size

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (axis1, axis2), angle = ellipse

            major = max(axis1, axis2)
            minor = min(axis1, axis2)
            aspect_ratio = major / max(minor, 1)

            # Reject extremely elongated regions
            if aspect_ratio > 3.5:
                continue
        else:
            continue

        equivalent_diameter_km = 2 * np.sqrt(area / np.pi) * km_per_pixel

        detections.append({
            "contour": contour,
            "area_pixels": area,
            "equivalent_diameter_km": equivalent_diameter_km,
            "centre": (int(cx), int(cy)),
            "ellipse": ellipse,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio
        })

    CONTOUR_THICKNESS = 2 
    FONT_SCALE = 1.5
    TEXT_THICKNESS = 1

    for i, item in enumerate(detections):
        cv2.drawContours(
            result,
            [item["contour"]],
            -1,
            (0, 255, 0),
            CONTOUR_THICKNESS
        )

        cx, cy = item["centre"]

        cv2.putText(
            result,
            f"Basin {i + 1}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 255, 255),
            TEXT_THICKNESS
        )

    if verbose:
        print("Potential basin regions:", len(detections))
        for i, item in enumerate(detections):
            print(
                f"Basin {i + 1}: "
                f"Area = {item['area_pixels']:.0f} pixels, "
                f"Equivalent diameter = {item['equivalent_diameter_km']:.1f} km, "
                f"Circularity = {item['circularity']:.2f}"
            )

    if detection_output_path:
        os.makedirs(os.path.dirname(detection_output_path) or ".", exist_ok=True)
        cv2.imwrite(detection_output_path, result)

    return detections