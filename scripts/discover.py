"""Scan a CAN bus for Robstride actuators by pinging device IDs 1..50."""

import argparse

from actuator_control import RobstrideBus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="can0")
    args = p.parse_args()

    found = []
    for device_id in range(1, 51):
        response = RobstrideBus.ping_by_id(args.channel, device_id, timeout=0.1)
        if response is None:
            continue
        print(f"  id={device_id}: {response}")
        found.append(device_id)
    print(f"\nFound {len(found)} actuator(s) on {args.channel}.")


if __name__ == "__main__":
    main()
