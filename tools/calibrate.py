"""
Interactive calibration for a camera: counting line, scale, and region of interest.

WHY THIS TOOL EXISTS:
Two values in config.yaml cannot be guessed, and both are currently placeholders:

  counting_line     where vehicles are counted. Flow rate - the number the signal
                    controller reacts to and the LSTM is trained on - is
                    meaningless until this sits across the actual stop line.
  pixels_per_meter  the scale. Without it every speed is reported as
                    "uncalibrated", because pixel motion cannot be converted to
                    km/h without knowing how big a pixel is.

Typing pixel coordinates by hand means opening the video, hovering over it in a
player and guessing. This tool lets you click them.

USAGE:
    python tools/calibrate.py data/sample_traffic.mp4

    l  line mode   - click the two ends of the counting line (across the road,
                     at the stop line)
    s  scale mode  - click two points a KNOWN real distance apart, then type
                     that distance in metres at the terminal
    r  roi mode    - click two corners of the region to detect within
    n / b          - step forward / back through the video to find a frame with
                     clear traffic
    u              - undo the last click
    p              - print the YAML block so far
    q              - quit and print the final YAML

WHAT TO MEASURE FOR SCALE - READ THIS, IT IS COUNTERINTUITIVE:

  MEASURE ALONG THE DIRECTION VEHICLES TRAVEL. Not across the road.

  This is the single easiest thing to get wrong here, and it fails silently.
  A camera looking down a road has TWO different scales:

     lateral       across the frame, e.g. lane width
     longitudinal  into the frame, along the road

  Perspective compresses the longitudinal axis far more than the lateral one.
  On the sample footage this project was developed against, lane width gave
  about 53 px/m while the along-the-road scale at the same height was about
  7.5 px/m - a factor of seven.

  Vehicles move longitudinally. So calibrating on lane width and applying it to
  motion produced speeds of 4.7 km/h on a motorway where traffic was doing 100.
  Nothing errored. The number was just wrong by 7x.

  Good things to measure, all ALONG the road, near the counting line:
  1. A painted lane divider plus one gap - the most reliable, because the
     spacing is standardised. In India IRC 35 specifies 3m mark + 6m gap (9m
     per cycle) for centre lines on rural highways; German Autobahn is 6m + 12m.
     Confirm against the standard for your road type.
  2. The gap between two roadside posts or lamp standards, if you can measure
     or look up the spacing.
  3. A vehicle of known length, measured front to back along its travel
     direction - a Maruti Alto is 3.4m, an auto-rickshaw about 2.6m. Roughest,
     but better than a lateral measurement.

  Measure inside the speed zone - the band around the counting line - never
  near the horizon. The longitudinal scale changes with every metre of depth,
  so one number is only an approximation, valid in a narrow band.

  THE PROPER FIX, if speed accuracy matters for your report: a homography.
  Click four points on the road surface that form a known real-world rectangle,
  compute a perspective transform to a bird's-eye view with cv2.findHomography,
  and measure distances there instead. That gives correct metres everywhere on
  the road plane rather than approximately-correct metres in one band. It is
  about forty lines and is the standard approach in published traffic work.

WHY THIS PRINTS YAML RATHER THAN EDITING config.yaml:
config.yaml is heavily commented, and every YAML library available here
(PyYAML included) discards comments on rewrite. Rewriting the file would
silently strip the documentation that makes it usable. So the tool prints a
block and you paste it - your comments survive.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

MAX_DISPLAY_WIDTH = 1280        # downscale for display; clicks are mapped back

MODE_LABELS = {
    "line": "COUNTING LINE  - click the two ends, across the road",
    "scale": "SCALE          - click two points ALONG the road (not across it)",
    "roi": "REGION         - click two opposite corners",
}
MODE_COLORS = {
    "line": (0, 165, 255),      # orange
    "scale": (0, 255, 255),     # yellow
    "roi": (0, 255, 0),         # green
}


class Calibrator:
    def __init__(self, source: str, camera_name: str = "my_camera"):
        self.source = source
        self.camera_name = camera_name

        self.capture = cv2.VideoCapture(source if not source.isdigit() else int(source))
        if not self.capture.isOpened():
            raise SystemExit(
                f"Could not open '{source}'.\n"
                f"  - for a file, check the path is right relative to where you are running from\n"
                f"  - for a webcam, pass 0"
            )

        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_index = 0
        self.frame = None
        self._read_frame(0)

        h, w = self.frame.shape[:2]
        self.width, self.height = w, h
        # Scale only downwards - never blow up a small frame, it just blurs.
        self.scale = min(1.0, MAX_DISPLAY_WIDTH / w)

        self.mode = "line"
        self.points: dict = {"line": [], "scale": [], "roi": []}
        self.meters: Optional[float] = None

    # ---- video -------------------------------------------------------

    def _read_frame(self, index: int) -> None:
        if self.total_frames:
            index = max(0, min(index, self.total_frames - 1))
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if ok:
            self.frame = frame
            self.frame_index = index
        elif self.frame is None:
            raise SystemExit(f"Could not read any frame from '{self.source}'.")

    def step(self, delta: int) -> None:
        self._read_frame(self.frame_index + delta)

    # ---- interaction -------------------------------------------------

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Map the click from display coordinates back to true frame pixels.
        # Forgetting this is the classic bug here: on a 4K video every
        # coordinate ends up a third of where you clicked.
        real = (int(x / self.scale), int(y / self.scale))
        bucket = self.points[self.mode]
        if len(bucket) >= 2:
            bucket.clear()
        bucket.append(real)
        print(f"  {self.mode}: point {len(bucket)} at {real}")
        if self.mode == "scale" and len(bucket) == 2:
            self._ask_distance()

    def _ask_distance(self) -> None:
        (x1, y1), (x2, y2) = self.points["scale"]
        pixels = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        print(f"\n  That span is {pixels:.1f} pixels.")
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > dy * 1.5:
            print("  WARNING: that measurement runs ACROSS the frame, not along the road.")
            print("           Lateral scale is not longitudinal scale - see the note at the")
            print("           top of this file. Speeds will come out several times too low.")
        try:
            raw = input("  How many METRES is it in reality? (e.g. 9 for one dash+gap) > ").strip()
            self.meters = float(raw)
            if self.meters <= 0:
                raise ValueError
            print(f"  -> pixels_per_meter = {pixels / self.meters:.2f}\n")
        except (ValueError, EOFError):
            self.meters = None
            print("  Not a valid number - scale discarded, click two points again.\n")

    def undo(self) -> None:
        bucket = self.points[self.mode]
        if bucket:
            print(f"  undid {self.mode} point at {bucket.pop()}")

    # ---- drawing -----------------------------------------------------

    def render(self):
        canvas = self.frame.copy()

        roi = self.points["roi"]
        if len(roi) == 2:
            (x1, y1), (x2, y2) = roi
            cv2.rectangle(canvas, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)),
                          MODE_COLORS["roi"], 2)
            cv2.putText(canvas, "ROI", (min(x1, x2) + 5, min(y1, y2) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, MODE_COLORS["roi"], 2)

        line = self.points["line"]
        if len(line) == 2:
            cv2.line(canvas, line[0], line[1], MODE_COLORS["line"], 3)
            cv2.putText(canvas, "COUNT", (line[0][0] + 5, line[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, MODE_COLORS["line"], 2)

        scale_pts = self.points["scale"]
        if len(scale_pts) == 2:
            cv2.line(canvas, scale_pts[0], scale_pts[1], MODE_COLORS["scale"], 2)
            label = f"{self.meters:.2f} m" if self.meters else "? m"
            cv2.putText(canvas, label, scale_pts[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, MODE_COLORS["scale"], 2)

        for pts, colour in ((line, MODE_COLORS["line"]),
                            (scale_pts, MODE_COLORS["scale"]),
                            (roi, MODE_COLORS["roi"])):
            for point in pts:
                cv2.circle(canvas, point, 6, colour, -1)

        banner = MODE_LABELS[self.mode]
        frame_info = f"frame {self.frame_index}/{self.total_frames or '?'}"
        cv2.rectangle(canvas, (0, 0), (self.width, 70), (0, 0, 0), -1)
        cv2.putText(canvas, banner, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, MODE_COLORS[self.mode], 2)
        cv2.putText(canvas, f"[l]ine [s]cale [r]oi  [n]ext [b]ack  [u]ndo  [p]rint  [q]uit   {frame_info}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if self.scale < 1.0:
            canvas = cv2.resize(canvas, (int(self.width * self.scale), int(self.height * self.scale)))
        return canvas

    # ---- output ------------------------------------------------------

    def pixels_per_meter(self) -> Optional[float]:
        pts = self.points["scale"]
        if len(pts) != 2 or not self.meters:
            return None
        (x1, y1), (x2, y2) = pts
        return round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / self.meters, 2)

    def yaml_block(self) -> str:
        line = self.points["line"]
        roi = self.points["roi"]
        ppm = self.pixels_per_meter()

        if len(roi) == 2:
            (rx1, ry1), (rx2, ry2) = roi
            roi_block = (f"    roi:\n"
                         f"      x1: {min(rx1, rx2)}\n"
                         f"      y1: {min(ry1, ry2)}\n"
                         f"      x2: {max(rx1, rx2)}\n"
                         f"      y2: {max(ry1, ry2)}")
        else:
            roi_block = (f"    roi:\n      x1: 0\n      y1: 0\n"
                         f"      x2: {self.width}\n      y2: {self.height}")

        line_value = (f"[[{line[0][0]}, {line[0][1]}], [{line[1][0]}, {line[1][1]}]]"
                      if len(line) == 2 else "# NOT SET - run again and press 'l'")
        ppm_value = f"{ppm}" if ppm else "null    # not calibrated; speeds stay uncalibrated"

        source = self.source.replace("\\", "/")
        return (
            f"  {self.camera_name}:\n"
            f"    id: \"{self.camera_name}\"\n"
            f"    name: \"{self.camera_name.replace('_', ' ').title()}\"\n"
            f"    source: \"{source}\"\n"
            f"    direction: \"north\"        # adjust to the approach this camera watches\n"
            f"{roi_block}\n"
            f"    counting_line: {line_value}\n"
            f"    pixels_per_meter: {ppm_value}\n"
            f"    fps: 30\n"
        )

    def print_yaml(self) -> None:
        print("\n" + "=" * 68)
        print("Paste this under 'cameras:' in config/config.yaml")
        print("=" * 68)
        print(self.yaml_block())
        print("=" * 68)
        if not self.points["line"]:
            print("WARNING: no counting line set. Flow rate will be meaningless.")
        if not self.pixels_per_meter():
            print("NOTE: no scale set. Speeds will be reported as uncalibrated,")
            print("      which is correct and honest - everything else still works.")
        print()

    # ---- loop --------------------------------------------------------

    def run(self) -> None:
        window = "Calibrate - see terminal for instructions"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.on_mouse)

        print(f"\nCalibrating '{self.source}'  ({self.width}x{self.height})")
        print("Click points in the window. Keys: l/s/r switch mode, n/b step frames,")
        print("u undo, p print, q quit.\n")

        while True:
            cv2.imshow(window, self.render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("l"):
                self.mode = "line"
            elif key == ord("s"):
                self.mode = "scale"
            elif key == ord("r"):
                self.mode = "roi"
            elif key == ord("n"):
                self.step(15)
            elif key == ord("b"):
                self.step(-15)
            elif key == ord("u"):
                self.undo()
            elif key == ord("p"):
                self.print_yaml()

        cv2.destroyAllWindows()
        self.capture.release()
        self.print_yaml()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Click the counting line and scale for a camera, and get YAML to paste."
    )
    parser.add_argument("source", help="video file path, or a webcam index like 0")
    parser.add_argument("--name", default="my_camera",
                        help="camera key to use in the generated YAML (default: my_camera)")
    args = parser.parse_args()

    if not args.source.isdigit() and not Path(args.source).exists():
        raise SystemExit(f"File not found: {args.source}")

    Calibrator(args.source, args.name).run()


if __name__ == "__main__":
    main()
