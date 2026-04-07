import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DepthStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.depth_frame: Optional[np.ndarray] = None
        self.header = None
        self.last_update_time = 0.0

    def update(self, frame: np.ndarray, header):
        with self.lock:
            self.depth_frame = frame.copy()
            self.header = header
            self.last_update_time = time.time()

    def get_frame(self):
        with self.lock:
            if self.depth_frame is None:
                return None, None, 0.0
            return self.depth_frame.copy(), self.header, self.last_update_time


depth_store = DepthStore()


class DepthRelaySubscriber(Node):
    def __init__(self):
        super().__init__('depth_http_server_node')

        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            '/depth_relay',
            self.depth_callback,
            qos
        )

        self.get_logger().info('Subscribed to /depth_relay')

    def depth_callback(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # 期望是 uint16 + 单通道
            if depth_img.dtype != np.uint16:
                self.get_logger().warn(f'Unexpected dtype: {depth_img.dtype}')
                return

            if len(depth_img.shape) != 2:
                self.get_logger().warn(f'Unexpected shape: {depth_img.shape}')
                return

            depth_store.update(depth_img, msg.header)

        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')


app = FastAPI(title="Depth HTTP Server")


@app.get("/health")
def health():
    frame, header, ts = depth_store.get_frame()
    if frame is None:
        return {
            "ok": False,
            "message": "No depth frame received yet"
        }

    h, w = frame.shape
    return {
        "ok": True,
        "width": w,
        "height": h,
        "dtype": str(frame.dtype),
        "frame_id": header.frame_id if header else "",
        "last_update_age_sec": round(time.time() - ts, 3)
    }


@app.get("/depth_at")
def depth_at(
    u: int = Query(..., description="x pixel"),
    v: int = Query(..., description="y pixel"),
    window: int = Query(2, ge=0, le=20, description="half window size")
):
    frame, header, ts = depth_store.get_frame()
    if frame is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": "No depth frame available"}
        )

    h, w = frame.shape
    if not (0 <= u < w and 0 <= v < h):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": f"Pixel out of range: ({u}, {v}), image size=({w}, {h})"}
        )

    x1 = max(0, u - window)
    y1 = max(0, v - window)
    x2 = min(w, u + window + 1)
    y2 = min(h, v + window + 1)

    patch = frame[y1:y2, x1:x2]
    valid = patch[patch > 0]

    center_value = int(frame[v, u])

    if valid.size == 0:
        return {
            "ok": True,
            "u": u,
            "v": v,
            "center_depth": center_value,
            "depth_median": 0,
            "valid_pixels": 0,
            "unit": "usually_mm",
            "frame_id": header.frame_id if header else "",
            "age_sec": round(time.time() - ts, 3)
        }

    depth_median = int(np.median(valid))
    depth_mean = float(np.mean(valid))

    return {
        "ok": True,
        "u": u,
        "v": v,
        "center_depth": center_value,
        "depth_median": depth_median,
        "depth_mean": round(depth_mean, 2),
        "valid_pixels": int(valid.size),
        "window": window,
        "unit": "usually_mm",
        "frame_id": header.frame_id if header else "",
        "age_sec": round(time.time() - ts, 3)
    }


@app.get("/depth_box")
def depth_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int
):
    frame, header, ts = depth_store.get_frame()
    if frame is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": "No depth frame available"}
        )

    h, w = frame.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 < x1 or y2 < y1:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": "Invalid box coordinates"}
        )

    roi = frame[y1:y2 + 1, x1:x2 + 1]
    valid = roi[roi > 0]

    if valid.size == 0:
        return {
            "ok": True,
            "box": [x1, y1, x2, y2],
            "depth_median": 0,
            "depth_mean": 0,
            "depth_min": 0,
            "depth_max": 0,
            "valid_pixels": 0,
            "unit": "usually_mm",
            "frame_id": header.frame_id if header else "",
            "age_sec": round(time.time() - ts, 3)
        }

    return {
        "ok": True,
        "box": [x1, y1, x2, y2],
        "depth_median": int(np.median(valid)),
        "depth_mean": round(float(np.mean(valid)), 2),
        "depth_min": int(np.min(valid)),
        "depth_max": int(np.max(valid)),
        "valid_pixels": int(valid.size),
        "unit": "usually_mm",
        "frame_id": header.frame_id if header else "",
        "age_sec": round(time.time() - ts, 3)
    }


@app.get("/depth_frame_png")
def depth_frame_png():
    frame, header, ts = depth_store.get_frame()
    if frame is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": "No depth frame available"}
        )

    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": "Failed to encode PNG"}
        )

    from fastapi.responses import Response
    return Response(content=encoded.tobytes(), media_type="image/png")


def ros_spin():
    rclpy.init()
    node = DepthRelaySubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8090)