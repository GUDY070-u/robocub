import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


YOLO_WEIGHT_PATH = "/home/user/Desktop/yolov11/runs/buplo/train1_05_04/weights/best.pt"
model = YOLO(YOLO_WEIGHT_PATH)


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, 300)


spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

USE_HOLE_FILLING = False
hole_filling = rs.hole_filling_filter()


def get_color_mask(color_img):
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    blue = cv2.inRange(hsv, (90, 70, 40), (130, 255, 255))
    green = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    yellow = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))

    mask = cv2.bitwise_or(red, blue)
    mask = cv2.bitwise_or(mask, green)
    mask = cv2.bitwise_or(mask, yellow)

    return mask


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_depth_edge(depth_m, object_mask):
    depth_valid = depth_m.copy()
    depth_valid[depth_valid <= 0] = 0

    depth_norm = cv2.normalize(
        depth_valid,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    depth_norm = cv2.bitwise_and(
        depth_norm,
        depth_norm,
        mask=object_mask
    )

    depth_blur = cv2.GaussianBlur(depth_norm, (5, 5), 0)
    depth_edge = cv2.Canny(depth_blur, 30, 90)

    return depth_edge


def estimate_face_state(roi_depth, object_mask):
    """
    바닥 대비 물체 높이로 윗면 / 뒷면 구분
    BACK : 약 0.5 cm
    TOP  : 약 2.4 cm
    """

    obj_depth_values = roi_depth[object_mask > 0]
    obj_depth_values = obj_depth_values[obj_depth_values > 0]

    if len(obj_depth_values) < 30:
        return "UNKNOWN", 0.0, 0.0, 0.0

    # 물체가 아닌 ROI 영역을 바닥 후보로 사용
    floor_mask = np.ones_like(object_mask, dtype=np.uint8)
    floor_mask[object_mask > 0] = 0

    floor_depth_values = roi_depth[floor_mask > 0]
    floor_depth_values = floor_depth_values[floor_depth_values > 0]

    if len(floor_depth_values) < 30:
        return "UNKNOWN", 0.0, 0.0, 0.0

    obj_z = np.median(obj_depth_values)
    floor_z = np.median(floor_depth_values)

    # 카메라에서 가까운 물체일수록 depth가 작음
    height = floor_z - obj_z

    if height < 0.012:
        face_state = "BACK"
    elif height > 0.015:
        face_state = "TOP"
    else:
        face_state = "UNKNOWN"

    return face_state, height, obj_z, floor_z


try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        if USE_HOLE_FILLING:
            depth_frame = hole_filling.process(depth_frame)

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_m = depth_img.astype(np.float32) * depth_scale

        yolo_result = model(
            color_img,
            conf=0.5,
            verbose=False
        )[0]

        result = color_img.copy()

        for box_data in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])

            conf = float(box_data.conf[0])
            cls_id = int(box_data.cls[0])
            class_name = model.names[cls_id]

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            padding = 20

            x1p = max(0, x1 - padding)
            y1p = max(0, y1 - padding)
            x2p = min(color_img.shape[1], x2 + padding)
            y2p = min(color_img.shape[0], y2 + padding)

            roi_color = color_img[y1p:y2p, x1p:x2p]
            roi_depth = depth_m[y1p:y2p, x1p:x2p]

            color_mask = get_color_mask(roi_color)

            depth_mask = np.logical_and(
                roi_depth > 0.15,
                roi_depth < 1.2
            )
            depth_mask = depth_mask.astype(np.uint8) * 255

            object_mask = cv2.bitwise_and(color_mask, depth_mask)
            object_mask = clean_mask(object_mask)

            face_state, height, obj_z, floor_z = estimate_face_state(
                roi_depth,
                object_mask
            )

            color_edge = cv2.Canny(object_mask, 50, 150)
            depth_edge = get_depth_edge(roi_depth, object_mask)

            combined_edge = cv2.bitwise_or(color_edge, depth_edge)
            combined_edge = cv2.bitwise_and(combined_edge, object_mask)

            contours, _ = cv2.findContours(
                object_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area < 500:
                    continue

                rect = cv2.minAreaRect(cnt)

                box_pts = cv2.boxPoints(rect)
                box_pts = np.intp(box_pts)

                box_pts[:, 0] += x1p
                box_pts[:, 1] += y1p

                cx = int(rect[0][0]) + x1p
                cy = int(rect[0][1]) + y1p

                angle = rect[2]

                if cx < 0 or cy < 0 or cx >= depth_m.shape[1] or cy >= depth_m.shape[0]:
                    continue

                z = depth_m[cy, cx]

                if z <= 0:
                    continue

                pixel_x = cx
                pixel_y = cy

                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrin,
                    [pixel_x, pixel_y],
                    z
                )

                X = point_3d[0]
                Y = point_3d[1]
                Z = point_3d[2]

                cv2.drawContours(
                    result,
                    [box_pts],
                    0,
                    (0, 255, 255),
                    2
                )

                for p in box_pts:
                    px, py = p
                    cv2.circle(
                        result,
                        (px, py),
                        5,
                        (0, 0, 255),
                        -1
                    )

                cv2.circle(
                    result,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )

                # ====================================
                # 바운딩박스 외부 텍스트 배치
                # ====================================
                line_gap = 25

                info_x = x1
                info_y = max(30, y1 - 105)

                cv2.putText(
                    result,
                    f"{class_name} {conf:.2f}",
                    (info_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    result,
                    f"pixel: ({pixel_x}, {pixel_y})",
                    (info_x, info_y + line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    result,
                    f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}",
                    (info_x, info_y + line_gap * 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    result,
                    f"face: {face_state}",
                    (info_x, info_y + line_gap * 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    result,
                    f"h:{height*100:.1f}cm obj:{obj_z:.3f} floor:{floor_z:.3f}",
                    (info_x, info_y + line_gap * 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                angle_x = x1
                angle_y = min(result.shape[0] - 10, y2 + 35)

                cv2.putText(
                    result,
                    f"angle: {angle:.1f} deg",
                    (angle_x, angle_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.imshow("ROI Color", roi_color)
            cv2.imshow("ROI Mask", object_mask)
            cv2.imshow("ROI Edge", combined_edge)

        cv2.imshow("YOLO + Color + Depth Edge", result)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()