# PianoRollReader

## Score editing and export

Right-click a measure and choose **Edit bar**. Select a note or rest to choose
the hand and edit target. **Edit LilyPond for selected hand** opens the
LilyPond editor for that hand and measure; it supports the generated note,
chord, rest, duration and tie syntax on the sixteenth-note grid, not arbitrary
LilyPond commands. Add several edits to the list or apply the pending edit with
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

While LilyPond renders, the score dialog shows an activity bar (not a
percentage). Rendering may be cancelled and retried without discarding the
edited notes. Reopening an unchanged score during the same session reuses the
rendered result; opening a project in a new session requires rendering again.

**Open project** restores the saved editing state. The live detection tracker is
not stored, so processing cannot be resumed from the saved frame. Play and
Step are disabled after opening; **Restart** starts a fresh detection and
discards the loaded detection and editing state. Save the project first.
