import argparse
import time
from pathlib import Path

import cv2
import mido
import numpy as np

from pianorollreader import (
	DEFAULT_VIDEO,
	find_frame_zero_keys,
	midi_note_name,
	save_frame_zero_baseline,
)


def prepare_key_masks(
	keys: list[dict[str, object]], key_width: float
) -> list[np.ndarray]:
	kernel_size = max(3, round(key_width * 0.28) | 1)
	shrink_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
	masks = []
	for key in keys:
		x, y, width, height = key["bbox"]
		contour = np.array(key["contour"], dtype=np.int32).reshape(-1, 1, 2)
		local_contour = contour - (x, y)
		mask = np.zeros((height, width), dtype=np.uint8)
		cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
		mask = cv2.erode(mask, shrink_kernel)
		masks.append(mask > 0)
	return masks


def score_frame(
	baseline: np.ndarray,
	frame: np.ndarray,
	keys: list[dict[str, object]],
	masks: list[np.ndarray],
) -> dict[str, str]:
	pressed: dict[str, str] = {}
	for key, interior_mask in zip(keys, masks):
		x, y, width, height = key["bbox"]
		mask_pixel_count = np.count_nonzero(interior_mask)
		if not mask_pixel_count:
			continue

		frame_region = frame[y : y + height, x : x + width]
		baseline_region = baseline[y : y + height, x : x + width]
		region_hsv = cv2.cvtColor(frame_region, cv2.COLOR_BGR2HSV)
		color_delta = cv2.absdiff(frame_region, baseline_region).max(axis=2)
		changed_region = (region_hsv[:, :, 1] > 100) & (color_delta > 80)
		interior_changes = changed_region & interior_mask
		if np.count_nonzero(interior_changes) / mask_pixel_count < 0.55:
			continue

		hues = region_hsv[:, :, 0][interior_changes]
		cyan_fraction = float(((hues >= 85) & (hues <= 110)).mean())
		green_fraction = float(((hues >= 45) & (hues <= 80)).mean())
		if max(cyan_fraction, green_fraction) < 0.5:
			continue
		pressed[str(key["note"])] = (
			"left" if cyan_fraction > green_fraction else "right"
		)
	return pressed


def collect_events(
	capture: cv2.VideoCapture,
	baseline: np.ndarray,
	keys: list[dict[str, object]],
	masks: list[np.ndarray],
	fps: float,
	frame_count: int,
	progress_interval: int,
) -> list[dict[str, object]]:
	events: list[dict[str, object]] = []
	active: dict[str, dict[str, object]] = {}
	started = time.perf_counter()
	frame_number = 0

	while True:
		ok, frame = capture.read()
		if not ok:
			break

		pressed = score_frame(baseline, frame, keys, masks)
		for note in list(active):
			if note not in pressed:
				event = active.pop(note)
				event["offset_frame"] = frame_number
				event["offset_seconds"] = frame_number / fps
				event["duration_seconds"] = (
					event["offset_seconds"] - event["onset_seconds"]
				)

		for note, hand in pressed.items():
			if note not in active:
				event = {
					"event_id": len(events) + 1,
					"note": note,
					"midi_note": next(
						number
						for number in range(21, 109)
						if midi_note_name(number) == note
					),
					"hand": hand,
					"onset_frame": frame_number,
					"offset_frame": None,
					"onset_seconds": frame_number / fps,
					"offset_seconds": None,
					"duration_seconds": None,
				}
				events.append(event)
				active[note] = event

		frame_number += 1
		if frame_number % progress_interval == 0:
			elapsed = time.perf_counter() - started
			throughput = frame_number / elapsed if elapsed else 0.0
			percentage = 100 * frame_number / frame_count if frame_count else 0.0
			print(
				f"{frame_number}/{frame_count} frames ({percentage:.1f}%), "
				f"{throughput:.1f} fps, {len(events)} hits, "
				f"{elapsed:.1f}s elapsed",
				flush=True,
			)

	end_seconds = frame_number / fps
	for event in active.values():
		event["offset_frame"] = frame_number
		event["offset_seconds"] = end_seconds
		event["duration_seconds"] = end_seconds - event["onset_seconds"]
	return events


def export_midi(events: list[dict[str, object]], fps: float, output_path: Path) -> None:
	tempo = 500_000
	ticks_per_beat = 480
	midi_file = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

	for channel, hand in enumerate(("left", "right")):
		track = mido.MidiTrack()
		midi_file.tracks.append(track)
		track.append(mido.MetaMessage("track_name", name=f"{hand.title()} hand"))
		if channel == 0:
			track.append(mido.MetaMessage("set_tempo", tempo=tempo))
		track.append(mido.Message("program_change", channel=channel, program=0))

		absolute_events = []
		for event in events:
			if event["hand"] != hand:
				continue
			onset_tick = round(
				mido.second2tick(
					float(event["onset_frame"]) / fps,
					ticks_per_beat,
					tempo,
				)
			)
			offset_tick = round(
				mido.second2tick(
					float(event["offset_frame"]) / fps,
					ticks_per_beat,
					tempo,
				)
			)
			offset_tick = max(onset_tick + 1, offset_tick)
			note = int(event["midi_note"])
			absolute_events.append((onset_tick, 1, note, "note_on"))
			absolute_events.append((offset_tick, 0, note, "note_off"))

		previous_tick = 0
		for tick, _, note, event_type in sorted(absolute_events):
			track.append(
				mido.Message(
					event_type,
					channel=channel,
					note=note,
					velocity=90 if event_type == "note_on" else 0,
					time=tick - previous_tick,
				)
			)
			previous_tick = tick
		track.append(mido.MetaMessage("end_of_track"))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	midi_file.save(str(output_path))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Detect piano key hits from a video without opening a GUI."
	)
	parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--progress-interval", type=int, default=1000)
	args = parser.parse_args()
	if args.progress_interval < 1:
		parser.error("--progress-interval must be positive")

	capture = cv2.VideoCapture(str(args.video))
	if not capture.isOpened():
		raise SystemExit(f"Could not open video: {args.video}")
	fps = capture.get(cv2.CAP_PROP_FPS)
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	if not np.isfinite(fps) or fps <= 0:
		capture.release()
		raise SystemExit("Video does not report a valid frame rate")

	ok, baseline = capture.read()
	if not ok:
		capture.release()
		raise SystemExit("Could not read frame 0")
	baseline_path = args.video.with_name(f"{args.video.stem}_key_contours.json")
	keys, key_width = save_frame_zero_baseline(baseline, args.video, baseline_path)
	masks = prepare_key_masks(keys, key_width)
	capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
	output_path = args.output or args.video.with_name(
		f"{args.video.stem}_headless.mid"
	)

	started = time.perf_counter()
	try:
		events = collect_events(
			capture,
			baseline,
			keys,
			masks,
			fps,
			frame_count,
			args.progress_interval,
		)
	finally:
		capture.release()

	export_midi(events, fps, output_path)
	elapsed = time.perf_counter() - started
	print(
		f"Finished {frame_count} frames in {elapsed:.1f}s; "
		f"{len(events)} key hits saved to {output_path}"
	)


if __name__ == "__main__":
	main()
