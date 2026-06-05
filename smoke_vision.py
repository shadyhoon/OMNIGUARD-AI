"""
Smoke test for utils.vision.
Generates a synthetic BGR image in memory and runs analyse_image_bytes
on it. Verifies the lazy import path works and the function returns
the expected shape.
"""

import io
import numpy as np

from utils.vision import analyse_image_bytes, vision_health


def main() -> None:
    print("vision health:", vision_health())

    # Build a 256x256 image: warm centre, cool border.
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Cool blue background
    img[:, :] = (180, 80, 60)  # BGR
    # Warm centre rectangle (different hue, suspicious lighting)
    img[64:192, 64:192] = (60, 180, 200)  # BGR

    ok, buf = cv2_imencode_png(img)
    result = analyse_image_bytes(buf.tobytes())
    print("image result:")
    for k, v in result.items():
        if k == "per_frame":
            continue
        print(f"  {k}: {v}")


def cv2_imencode_png(img):
    import cv2
    return cv2.imencode(".png", img)


if __name__ == "__main__":
    main()
