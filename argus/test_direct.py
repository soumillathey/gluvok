import argparse
import os
import time

from app.services import PlateRecognizerFactory
from app.services.image_processing import decode_and_downscale
from app.services.yolo_filter import filter_vehicle_and_occupancy

TESTS_DIR = "tests"


def parse_args():
    available_providers = [p.value for p in PlateRecognizerFactory.list_providers()]
    parser = argparse.ArgumentParser(description="ANPR Strategy Performance & Direct Testing CLI")
    parser.add_argument(
        "strategies",
        nargs="*",
        choices=available_providers,
        default=None,
        help=f"Recognition strategies to test ({', '.join(available_providers)}). Defaults to all.",
    )
    args = parser.parse_args()
    if not args.strategies:
        args.strategies = available_providers
    return args


def test_models():
    args = parse_args()

    image_paths = [
        os.path.join(TESTS_DIR, f) for f in os.listdir(TESTS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_paths.sort()

    selected_strategies = args.strategies
    print(f"Testing strategies: {', '.join(selected_strategies)}")

    strategy_engines = {}
    for st_name in selected_strategies:
        try:
            strategy_engines[st_name] = PlateRecognizerFactory.get_recognizer(st_name)
        except (ValueError, KeyError, RuntimeError, ImportError) as e:
            print(f"Error instantiating strategy '{st_name}': {e}")

    for img_path in image_paths:
        print(f"\n{'=' * 60}\nTesting image: {img_path}\n{'=' * 60}")

        with open(img_path, "rb") as f:
            raw_bytes = f.read()

        # Normalise + downscale exactly as the API does before passing to any strategy.
        # Previously non-PaddleOCR strategies received a raw file path, bypassing
        # decode_and_downscale and the YOLO box — this now matches the real code path.
        img_bytes = decode_and_downscale(raw_bytes)

        t_yolo_start = time.time()
        yolo_res = filter_vehicle_and_occupancy(img_bytes)
        t_yolo = round((time.time() - t_yolo_start) * 1000, 2)
        vehicle_box = yolo_res.get("vehicle_box")
        vehicle_boxes = yolo_res.get("vehicle_boxes")
        print(
            f"[YOLO v11 Prescreening] ({t_yolo:>7.2f} ms): vehicle={yolo_res['vehicle_type']}, box={vehicle_box}, count={yolo_res.get('vehicle_count', 1)}"
        )

        for st_name, engine in strategy_engines.items():
            t0 = time.time()
            result = engine.recognize(img_bytes, vehicle_box=vehicle_box, vehicle_boxes=vehicle_boxes)
            t_exec = round((time.time() - t0) * 1000, 2)
            print(f"[{st_name.upper():<20}] ({t_exec:>7.2f} ms): {result}")


if __name__ == "__main__":
    test_models()
