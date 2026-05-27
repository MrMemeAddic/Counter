# Face.py Refactor & Bug Fix Plan

## Information Gathered
- `Face.py` is a single-file people counter using YOLOv8, OpenCV, Pandas, and openpyxl.
- All logic runs at module level, making it impossible to unit-test without spawning a webcam window.
- Bugs identified:
  1. **Untestable structure**: No functions or `if __name__ == "__main__":` guard.
  2. **Memory leak / stale tracks**: `track_history` dict grows unbounded because old track IDs are never removed.
  3. **Counting semantics**: `entered_ids` and `exited_ids` are `set`s, so a re-entering person is never counted again.
  4. **Missing error handling**: No check if `cv2.VideoCapture` succeeds or if the YOLO model file exists.
  5. **Silent failure on no detections**: `results[0].boxes.id is not None` is correct, but if `results` is empty it could raise.

## Plan (file-level)
1. **Refactor `Face.py`**
   - Wrap everything into functions: `run_counter()`, `process_frame(frame, model, ...)`, `detect_crossing(...)`, `generate_reports(log_data)`.
   - Add `if __name__ == "__main__": run_counter()` guard.
   - Clear stale track IDs from `track_history` each frame (keep only current IDs).
   - Change counting to per-event (append to `log_data` every time a crossing occurs) instead of per-unique-ID.
   - Add a small startup check for video capture.
2. **Create `test_Face.py`**
   - Mock `cv2`, `ultralytics.YOLO`, `pandas.DataFrame`, `openpyxl.load_workbook`.
   - Test crossing logic (LEFT→RIGHT = ENTRY, RIGHT→LEFT = EXIT).
   - Test that reports are generated.
   - Ensure no unbounded memory growth.
3. **Run tests** with `pytest` (install if necessary) and fix any failures.
4. **Smoke-test** by running `python -c "import Face"` to confirm no syntax/import errors.

## Dependent Files
- `Face.py` (to be edited)
- `test_Face.py` (new file to be created)

## Follow-up Steps
- Install `pytest` if missing.
- Execute `pytest test_Face.py -v`.
- Fix any failing assertions.
