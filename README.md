# PianoRollReader

## Detection profiles

Open a video and choose **Detection profile** to adapt detection without
changing the built-in **Original** profile. Move the frame slider to a passage
with visible pressed keys and falling bars. Choose a sample mode and click a
clearly colored key or bar for each hand and key color (white or black); each
click suggests a separate hue range and, for keys, a minimum saturation;
both can also be edited manually. Pale colored keys can be sampled, but
near-gray pixels still cannot provide a reliable hand color. For
bar samples, choose the key color that the bar corresponds to. OpenCV hue
values are 0–179. The preview shows detected keys and waterfall segments.
Existing profiles use their original range for both key colors until changed.

Choose **Place keyboard line**, then click or drag on the keyboard's upper edge.
The yellow line marks that position and the red line is the waterfall tracking
line; adjust its pixel offset if necessary. The original fixed position remains
available. Check several frames and listen to the result: the preview shows
individual frames, not complete note onsets, offsets, or repeated hits.

Use **Save as...** to save a named profile for reuse, select it from the
profile list, and press **OK** to apply it. The Original profile cannot be
overwritten or deleted. Unsaved changes can also be applied to the current
video. Applying a changed profile restarts tracking; a confirmation warns
before existing detection results and timeline/score edits are discarded.
Save the current project first if you need to keep them.

For the Sacrifice video, a useful starting point is right-key and right-bar
hue ranges of 40–80 for both key colors, a keyboard position around 78.70%, and the default
18-pixel tracking offset. Verify and refine the settings on multiple passages.
Profiles adjust highlight and waterfall detection; keyboard key contours are
still calibrated from the first frame and require the expected 88-key layout.

## Score editing and export

Choose the score's key signature from the **Key signature** dropdown (major
and minor keys up to seven sharps or flats). Changing it on an existing score
re-renders the notation without regenerating notes from the source or
discarding edits, tuplets or reviewed measures. MIDI pitches do not change.
Spelling is based on the key signature rather than harmonic context; manual
enharmonic choices are not retained when rendering. Old projects default to
C major. Projects with a non-default key use version 6 and cannot be opened
by older releases.

The score BPM can be set from 20 to 300 before generating the score. For
normalized or quantized sources, their existing rhythmic positions are
preserved while their playback timing is scaled to the chosen score BPM.
Raw detection and editable-copy timings instead remain in seconds and are
interpreted on the chosen BPM grid. Normalized sources are still computed
using the 160 BPM reference; changing score BPM does not re-track the video
or change the normalization's underlying beat decisions.

Right-click a measure and choose **Edit bar**. Select a note or rest to choose
the hand and edit target. **Edit LilyPond for selected hand** opens the
LilyPond editor for that hand and measure; it supports the generated note,
chord, rest, duration and tie syntax, plus non-nested 3-, 5- and 7-tuplets,
not arbitrary LilyPond commands. For example, enter
`\tuplet 3/2 { c'8 d' e' } r2.` or `\times 2/3 { c'8 d' e' } r2.`.
The supported written durations range from whole notes to sixteenths;
tuplets may contain chords, rests and ties. A bar containing a tuplet must
total four beats. The normal editing actions remain available in other bars
and in the other hand. For a hand and bar containing tuplets, use LilyPond;
grid edits that would change or shift a tuplet in a later bar are rejected
to preserve its notation and timing. A tie ending a tuplet bar
must continue a note already held across the bar line.
Add several edits to the list or apply the pending edit with
OK. Insertion positions are zero-based sixteenths within the selected measure
(0–15).

Right-click a measure and choose **Mark bar as reviewed** to suppress its
uncertainty marker and skip it when navigating to uncertain measures. The
measured uncertainty values remain unchanged. The same menu can mark it
unchecked again. Reviewing is saved with the score in a project. If a later
note edit changes a reviewed measure, its review is cleared; generating a new
score clears all reviews.

**Save PDF, LilyPond and MIDI** exports the current score as a PDF, three
LilyPond files and a MIDI file with separate hand tracks. The MIDI uses the
score's current tempo and edited note timings.
After editing the title, leaving the title field updates it in the preview
without re-rendering the score. The preview uses an overlay, so its font may
differ from the final PDF. When the title has changed, exporting runs LilyPond
again to produce a PDF with the new title. Save the project to retain the title.

## Projects

Use **Save project** in the main window, timeline, or score dialog to save a
`.pianoroll` project. It contains the detected key and waterfall events, the
timeline's editable copy and quantization settings, and the current score's
edited notes and settings. Edits in the measure dialog must first be applied
with OK. The project stores a video path, not the video itself; if the video
has moved, select it when opening the project. LilyPond is required to
re-render a saved score.
Existing version-1 projects remain supported. Projects containing tuplets are
saved as version 2 so older releases reject them instead of silently losing
their rhythmic notation. Projects using a custom detection profile are saved
as version 3 and contain a copy of the profile, independent of the reusable
profiles stored in application settings. Older releases cannot open version 3.
Projects without a profile use the Original settings.
Projects with separate black-key color ranges use version 4; older releases
cannot open them. Previously saved version-3 profiles remain compatible.
Profiles with customized key saturation thresholds require project version 5.
Earlier versions cannot open these projects; older profiles keep the original
saturation threshold.

While LilyPond renders, the score dialog shows an activity bar (not a
percentage). Rendering may be cancelled and retried without discarding the
edited notes. Reopening an unchanged score during the same session reuses the
rendered result; opening a project in a new session requires rendering again.

**Open project** restores the saved editing state. The live detection tracker is
not stored, so processing cannot be resumed from the saved frame. Play and
Step are disabled after opening; **Restart** starts a fresh detection and
discards the loaded detection and editing state. Save the project first.
