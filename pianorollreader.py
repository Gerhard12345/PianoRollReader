import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEO = Path(
	r"Z:\Documents\Klaviernoten\Nikita\Elton John - Nikita (Live in Ephesus 2001) [Piano Tutorial].mp4"
)
WHITE_THRESHOLD = 215
DARK_PIXEL_THRESHOLD = 65
WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}
CHROMATIC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_note_name(midi_note: int) -> str:
	return f"{CHROMATIC_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


WHITE_NOTES = [21, 23] + [
	note for note in range(24, 109) if note % 12 in WHITE_PITCH_CLASSES
]
BLACK_NOTES = [
	note for note in range(21, 109) if note % 12 in BLACK_PITCH_CLASSES
]


def read_video_frame(video_path: Path, frame_number: int) -> np.ndarray:
	capture = cv2.VideoCapture(str(video_path))
	if not capture.isOpened():
		raise SystemExit(f"Could not open video: {video_path}")

	try:
		capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
		success, frame = capture.read()
	finally:
		capture.release()

	if not success:
		raise SystemExit(f"Could not read frame {frame_number} from: {video_path}")
	return frame


def find_frame_zero_keys(
	frame: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, float]:
	grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_height, frame_width = grayscale.shape
	keyboard_top = int(frame_height * 0.75)
	key_width = frame_width / 52

	white_band_height = max(1, round(key_width))
	white_band_top = frame_height - white_band_height
	white_band = grayscale[white_band_top:, :]
	white_band_mask = cv2.threshold(
		white_band, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY
	)[1]
	white_contours, _ = cv2.findContours(
		white_band_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)
	white_contours = sorted(
		(
			contour
			for contour in white_contours
			if 0.4 * key_width
			<= cv2.boundingRect(contour)[2]
		<= 1.6 * key_width
		),
		key=lambda contour: cv2.boundingRect(contour)[0],
	)

	black_band_height = max(1, round(key_width * 1.3))
	black_band_center = keyboard_top + round(key_width * 2.5)
	black_band_top = black_band_center - black_band_height // 2
	black_band_bottom = black_band_top + black_band_height
	black_band = grayscale[black_band_top:black_band_bottom, :]
	black_band_mask = cv2.threshold(
		black_band, DARK_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY_INV
	)[1]
	black_contours, _ = cv2.findContours(
		black_band_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)
	black_contours = sorted(
		(
			contour
			for contour in black_contours
			if 0.35 * key_width
			<= cv2.boundingRect(contour)[2]
			<= key_width
		),
		key=lambda contour: cv2.boundingRect(contour)[0],
	)
	if len(white_contours) != len(WHITE_NOTES) or len(black_contours) != len(
		BLACK_NOTES
	):
		raise SystemExit(
			"Frame 0 key contours did not match the expected keyboard: "
			f"found {len(white_contours)} white and {len(black_contours)} black"
		)

	white_mask = np.zeros_like(grayscale)
	white_mask[white_band_top:, :] = white_band_mask
	black_mask = np.zeros_like(grayscale)
	black_mask[black_band_top:black_band_bottom, :] = black_band_mask
	keys: list[dict[str, object]] = []
	for kind, contours, note_numbers, y_offset in (
		("white", white_contours, WHITE_NOTES, white_band_top),
		("black", black_contours, BLACK_NOTES, black_band_top),
	):
		for contour, midi_note in zip(contours, note_numbers):
			full_contour = contour + (0, y_offset)
			x, y, width, height = cv2.boundingRect(full_contour)
			keys.append(
				{
					"kind": kind,
					"note": midi_note_name(midi_note),
					"bbox": [x, y, width, height],
					"contour": full_contour.reshape(-1, 2).tolist(),
				}
			)
	return keys, white_mask, black_mask, key_width


def save_frame_zero_baseline(
	frame: np.ndarray, video_path: Path, baseline_path: Path
) -> tuple[list[dict[str, object]], float]:
	keys, white_mask, black_mask, key_width = find_frame_zero_keys(frame)
	baseline_path.parent.mkdir(parents=True, exist_ok=True)
	baseline_data = {
		"video": str(video_path),
		"frame": 0,
		"frame_size": [frame.shape[1], frame.shape[0]],
		"keys": keys,
	}
	baseline_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
	for suffix, mask in (("white_mask", white_mask), ("black_mask", black_mask)):
		mask_path = baseline_path.with_name(f"{baseline_path.stem}_{suffix}.png")
		if not cv2.imwrite(str(mask_path), mask):
			raise SystemExit(f"Could not save debug mask: {mask_path}")
	print(f"Saved frame 0 key contours: {baseline_path}")
	return keys, key_width


def score_key_highlights(
	baseline_frame: np.ndarray,
	target_frame: np.ndarray,
	keys: list[dict[str, object]],
	key_width: float,
) -> dict[str, str]:
	if baseline_frame.shape != target_frame.shape:
		raise SystemExit("Frame 0 and target frame have different dimensions.")

	baseline_hsv = cv2.cvtColor(baseline_frame, cv2.COLOR_BGR2HSV)
	target_hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)
	color_delta = np.abs(
		target_frame.astype(np.int16) - baseline_frame.astype(np.int16)
	).max(axis=2)
	changed_color = (target_hsv[:, :, 1] > 100) & (color_delta > 80)
	kernel_size = max(3, round(key_width * 0.28) | 1)
	shrink_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
	pressed: dict[str, str] = {}

	for key in keys:
		x, y, width, height = key["bbox"]
		contour = np.array(key["contour"], dtype=np.int32).reshape(-1, 1, 2)
		local_contour = contour - (x, y)
		interior_mask = np.zeros((height, width), dtype=np.uint8)
		cv2.drawContours(interior_mask, [local_contour], -1, 255, cv2.FILLED)
		interior_mask = cv2.erode(interior_mask, shrink_kernel)
		if not np.any(interior_mask):
			continue

		changed_region = changed_color[y : y + height, x : x + width]
		score = float(changed_region[interior_mask > 0].mean())
		if score >= 0.55:
			highlight_hues = target_hsv[y : y + height, x : x + width, 0][
				changed_region & (interior_mask > 0)
			]
			cyan_fraction = float(
				((highlight_hues >= 85) & (highlight_hues <= 110)).mean()
			)
			green_fraction = float(
				((highlight_hues >= 45) & (highlight_hues <= 80)).mean()
			)
			if max(cyan_fraction, green_fraction) < 0.5:
				continue
			hand = "left" if cyan_fraction > green_fraction else "right"
			pressed[str(key["note"])] = hand
	return pressed


def save_pressed_overlay(
	target_frame: np.ndarray,
	keys: list[dict[str, object]],
	pressed: dict[str, str],
	result_path: Path,
	show: bool,
	frame_number: int,
) -> None:
	result = target_frame.copy()
	for key in keys:
		note = str(key["note"])
		contour = np.array(key["contour"], dtype=np.int32).reshape(-1, 1, 2)
		active = note in pressed
		if not active:
			color = (150, 150, 150)
		elif key["kind"] == "white":
			color = (139, 0, 0)
		else:
			color = (0, 0, 139)
		cv2.drawContours(result, [contour], -1, color, 2)
		if active:
			x, y, _, _ = key["bbox"]
			cv2.putText(
				result,
				note,
				(x, max(15, y - 5)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.4,
				color,
				1,
				cv2.LINE_AA,
			)

	result_path.parent.mkdir(parents=True, exist_ok=True)
	if not cv2.imwrite(str(result_path), result):
		raise SystemExit(f"Could not save pressed-key overlay: {result_path}")
	print(f"Saved pressed-key overlay: {result_path}")
	if show:
		cv2.imshow(f"Pressed keys in frame {frame_number}", result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Use frame 0 key contours to detect pressed keys in a video frame."
	)
	parser.add_argument(
		"--frame",
		type=int,
		required=True,
		help="zero-based frame number to display",
	)
	parser.add_argument(
		"--video",
		type=Path,
		default=DEFAULT_VIDEO,
		help=f"video file (default: {DEFAULT_VIDEO})",
	)
	parser.add_argument(
		"--baseline",
		type=Path,
		default=Path("frame_0_keys.json"),
		help="path for the saved frame 0 contours",
	)
	parser.add_argument("--result", type=Path, help="path for the pressed-key overlay")
	parser.add_argument(
		"--no-display", action="store_true", help="save the result without opening a window"
	)
	args = parser.parse_args()
	if args.frame < 0:
		raise SystemExit("Frame number must be zero or greater.")

	frame_zero = read_video_frame(args.video, 0)
	frame_zero_keys, key_width = save_frame_zero_baseline(
		frame_zero, args.video, args.baseline
	)
	target_frame = read_video_frame(args.video, args.frame)
	pressed = score_key_highlights(
		frame_zero, target_frame, frame_zero_keys, key_width
	)
	note_positions = {
		str(key["note"]): int(key["bbox"][0]) for key in frame_zero_keys
	}
	for hand, color_name in (("left", "cyan"), ("right", "green")):
		hand_notes = sorted(
			(note for note, detected_hand in pressed.items() if detected_hand == hand),
			key=note_positions.__getitem__,
		)
		print(
			f"{hand.title()} hand ({color_name}): "
			f"{', '.join(hand_notes) if hand_notes else 'none'}"
		)
	if not pressed:
		print(f"No pressed keys detected in frame {args.frame}.")
	result_path = args.result or Path(f"frame_{args.frame}_pressed.png")
	save_pressed_overlay(
		target_frame,
		frame_zero_keys,
		pressed,
		result_path,
		not args.no_display,
		args.frame,
	)


if __name__ == "__main__":
	main()
