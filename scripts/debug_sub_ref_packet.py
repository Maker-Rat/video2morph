#!/usr/bin/env python3
"""Debug subscriber for live retargeting reference packets."""

import argparse
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Subscribe to delayed reference packets from video_to_go2_student_rt_live.py")
    p.add_argument("--sub", default="tcp://127.0.0.1:5555", help="ZMQ PUB endpoint to connect to")
    p.add_argument("--timeout-ms", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.sub)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))

    print(f"Subscribed to {args.sub}")
    last_seq = None
    try:
        while True:
            try:
                packet = sock.recv_pyobj()
            except zmq.Again:
                print("waiting...")
                continue

            refs = np.asarray(packet["refs"], dtype=np.float32)
            now = time.time()
            age_ms = (now - float(packet["timestamp"])) * 1000.0
            seq = int(packet["seq"])
            dropped = 0 if last_seq is None else max(0, seq - last_seq - 1)
            last_seq = seq
            print(
                f"seq={seq:06d} age={age_ms:6.1f}ms dropped={dropped:3d} "
                f"robot={packet.get('robot')} valid={packet.get('valid')} "
                f"shape={refs.shape} quat={packet.get('quat_convention')} "
                f"j0={refs[0, 0]: .3f} vx={refs[0, 12]: .3f} wz={refs[0, 17]: .3f}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(linger=0)


if __name__ == "__main__":
    main()
