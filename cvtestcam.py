#!/usr/bin/env python3
"""Probe OpenCV camera indices and print which ones deliver frames."""

import sys

import cv2


def backends():
    yield None, "default"
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        yield cv2.CAP_V4L2, "CAP_V4L2"
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        yield cv2.CAP_DSHOW, "CAP_DSHOW"
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        yield cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION"


def try_index(idx, backend):
    if backend is None:
        cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    ok, frame = cap.read()
    cap.release()
    return ok and frame is not None and frame.size > 0


def main():
    hi = 10
    if len(sys.argv) > 1:
        hi = int(sys.argv[1])

    print(f"Scanning camera indices 0..{hi}")
    any_ok = False
    for idx in range(hi + 1):
        for backend, bname in backends():
            if try_index(idx, backend):
                print(f"  OK  index={idx}  ({bname})")
                any_ok = True
                break

    if not any_ok:
        print("  (no working camera found)")
        sys.exit(1)


if __name__ == "__main__":
    main()
