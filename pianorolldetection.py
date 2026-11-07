"""Detection profile validation and key-highlight scoring."""

import copy

import cv2
import numpy as np

DEFAULT_PROFILE = {
	"name": "Original",
	"key_left": [85, 110],
	"key_right": [45, 80],
	"bar_left": [85, 110],
	"bar_right": [45, 80],
	"keyboard_fraction": None,
	"line_offset": 18,
}
HUE_FIELDS = ("key_left", "key_right", "bar_left", "bar_right")
BLACK_HUE_FIELDS = tuple(f"{field}_black" for field in HUE_FIELDS)
KEY_SATURATION_FIELDS = tuple(
	f"key_{hand}{kind}_saturation"
	for hand in ("left", "right") for kind in ("", "_black")
)
DEFAULT_KEY_SATURATION = 100


def validate_profile(profile):
	if (not isinstance(profile, dict) or not set(DEFAULT_PROFILE) <= set(profile)
			or set(profile) - set(DEFAULT_PROFILE) - set(BLACK_HUE_FIELDS) - set(KEY_SATURATION_FIELDS)):
		raise ValueError("Invalid detection profile fields.")
	if not isinstance(profile["name"], str) or not profile["name"].strip() or len(profile["name"]) > 100:
		raise ValueError("Profile name must contain 1 to 100 characters.")
	for key in HUE_FIELDS + tuple(field for field in BLACK_HUE_FIELDS if field in profile):
		value = profile[key]
		if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int for v in value) or not 0 <= value[0] <= value[1] <= 179:
			raise ValueError(f"Invalid hue range: {key}.")
	for key in KEY_SATURATION_FIELDS:
		if key in profile and (type(profile[key]) is not int or not 0 <= profile[key] <= 254):
			raise ValueError(f"Invalid minimum saturation: {key}.")
	value = profile["keyboard_fraction"]
	if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.1 <= value <= 0.99):
		raise ValueError("Keyboard position must be between 10% and 99% of frame height.")
	if type(profile["line_offset"]) is not int or not 0 <= profile["line_offset"] <= 200:
		raise ValueError("Strike-line offset must be between 0 and 200 pixels.")
	return copy.deepcopy(profile)


def hue_range(profile, field, kind):
	if kind == "black":
		return profile.get(f"{field}_black", profile[field])
	return profile[field]


def waterfall_hues(profile):
	white = tuple((hand, profile[f"bar_{hand}"]) for hand in ("left", "right"))
	black = tuple((hand, hue_range(profile, f"bar_{hand}", "black")) for hand in ("left", "right"))
	return white, black


def score_frame(frame, baseline, keys, masks, profile):
	if baseline is None or baseline.shape != frame.shape:
		return {}
	pressed = {}
	for key, interior_mask in zip(keys, masks):
		x, y, width, height = key["bbox"]
		mask_pixel_count = np.count_nonzero(interior_mask)
		if not mask_pixel_count:
			continue
		frame_region = frame[y:y + height, x:x + width]
		baseline_region = baseline[y:y + height, x:x + width]
		hsv = cv2.cvtColor(frame_region, cv2.COLOR_BGR2HSV)
		delta = cv2.absdiff(frame_region, baseline_region).max(axis=2)
		fractions = []
		for hand in ("left", "right"):
			field = f"key_{hand}" + ("_black" if key["kind"] == "black" else "")
			saturation = profile.get(f"{field}_saturation", DEFAULT_KEY_SATURATION)
			changed = (hsv[:, :, 1] > saturation) & (delta > 80) & interior_mask
			if np.count_nonzero(changed) / mask_pixel_count < 0.55:
				fractions.append(0.0)
				continue
			hues = hsv[:, :, 0][changed]
			low, high = hue_range(profile, f"key_{hand}", key["kind"])
			fractions.append(float(((hues >= low) & (hues <= high)).mean()))
		left_fraction, right_fraction = fractions
		if max(left_fraction, right_fraction) >= 0.5:
			pressed[str(key["note"])] = "left" if left_fraction > right_fraction else "right"
	return pressed
