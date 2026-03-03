import cv2
import time


def fourcc_to_str(v: float) -> str:
    code = int(v)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def open_and_probe(index: int, width: int = 640, height: int = 480, fps: int = 30):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, "open_failed"

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Warmup + probe whether this index can actually deliver frames
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            cap.release()
            return True, "ok"

    cap.release()
    return False, "read_failed"


def find_two_working_cameras(candidates=(0, 2, 1, 3, 4, 5, 6, 7)):
    working_indices = []
    print("开始探测摄像头索引...")
    for idx in candidates:
        ok, status = open_and_probe(idx)
        print(f"  index {idx}: {status}")
        if ok:
            working_indices.append(idx)
        if len(working_indices) >= 2:
            break
    return working_indices


def open_camera(index: int, width: int = 640, height: int = 480, fps: int = 30):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    # Try to keep driver queue shallow to reduce display latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # warmup
    for _ in range(8):
        cap.read()
    return cap


def main():
    cam_indices = find_two_working_cameras()
    if len(cam_indices) < 2:
        print("可用摄像头不足2个，请先检查索引、占用或权限。")
        return

    idx0, idx1 = cam_indices[0], cam_indices[1]
    cap0 = open_camera(idx0)
    cap1 = open_camera(idx1)
    if cap0 is None or cap1 is None:
        print("重新打开摄像头失败，请重试。")
        if cap0 is not None:
            cap0.release()
        if cap1 is not None:
            cap1.release()
        return

    print(f"使用摄像头 index {idx0} 和 {idx1}")
    print(
        f"index {idx0}: {int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"fps={cap0.get(cv2.CAP_PROP_FPS):.2f}, fourcc={fourcc_to_str(cap0.get(cv2.CAP_PROP_FOURCC))}"
    )
    print(
        f"index {idx1}: {int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"fps={cap1.get(cv2.CAP_PROP_FPS):.2f}, fourcc={fourcc_to_str(cap1.get(cv2.CAP_PROP_FOURCC))}"
    )

    fail0 = 0
    fail1 = 0
    while True:
        # Grab first for both cameras, then retrieve, to improve temporal alignment.
        g0 = cap0.grab()
        g1 = cap1.grab()
        t = time.time()
        ret0, frame0 = cap0.retrieve() if g0 else (False, None)
        ret1, frame1 = cap1.retrieve() if g1 else (False, None)
        if not ret0 or frame0 is None:
            fail0 += 1
        else:
            fail0 = 0
        if not ret1 or frame1 is None:
            fail1 += 1
        else:
            fail1 = 0
        if fail0 >= 10:
            print(f"连续读取失败: index {idx0}")
            break
        if fail1 >= 10:
            print(f"连续读取失败: index {idx1}")
            break
        if fail0 > 0 or fail1 > 0:
            continue

        frame0 = cv2.resize(frame0, (640, 480))
        frame1 = cv2.resize(frame1, (640, 480))
        cv2.putText(frame0, f"cam {idx0} {t:.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame1, f"cam {idx1} {t:.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        combined = cv2.hconcat([frame0, frame1])

        try:
            cv2.imshow(f"Camera {idx0} and Camera {idx1}", combined)
        except cv2.error:
            print("当前 OpenCV 不支持 GUI 显示（imshow），仅做读流测试后退出。")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
