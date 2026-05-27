# Counter
# 🚶 People Counter — Real-Time Entry/Exit Detection
A real-time people counter built with **YOLOv8** and **OpenCV** that tracks individuals crossing a configurable virtual line and generates detailed reports in CSV, XML, and Excel formats.
---
## ✨ Features
- **Real-time detection & tracking** — Uses YOLOv8n with DeepSORT-style persistent tracking to assign stable IDs to each person.
- **Movable counting line** — A vertical line that can be repositioned live using arrow keys or `A`/`D` keys.
- **Hysteresis buffer** — A configurable dead-zone around the counting line prevents false double-counts from jittery or slow walkers.
- **State-based crossing logic** — Direction is determined by committed side transitions (LEFT → RIGHT = ENTRY, RIGHT → LEFT = EXIT).
- **Live HUD overlay** — Displays FPS, IN/OUT counts, and per-person tracking IDs with styled bounding boxes.
- **Multi-format reporting** — On reset or exit, generates:
  - `people_log.csv` — Raw event log with TrackID, Type, and Timestamp.
  - `people_report.xlsx` — Formatted Excel report with an Entry vs. Exit pie chart.
  - `people_log.xml` — Structured XML log with a summary block.
- **Fullscreen toggle** — Press `F` to switch between windowed and fullscreen mode.
- **Comprehensive test suite** — 30+ unit tests covering all core logic using `pytest`.
---
## 📋 Requirements
- Python **3.9+**
- A webcam (or a video file)
### Python Dependencies
```text
ultralytics
opencv-python
numpy
pandas
openpyxl
```
Install all dependencies at once:
```bash
pip install ultralytics opencv-python numpy pandas openpyxl
```
> **Note:** The YOLOv8 nano model (`yolov8n.pt`) is included in the repository. Ultralytics will automatically download it if it is missing.
---
## 🚀 Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/people-counter.git
cd people-counter
```
### 2. Install Dependencies
```bash
pip install ultralytics opencv-python numpy pandas openpyxl
```
### 3. Run the Counter
```bash
python counter.py
```
By default, the app uses your webcam (`VIDEO_SOURCE = 0`). To use a video file instead, edit the top of `counter.py`:
```python
VIDEO_SOURCE = "path/to/your/video.mp4"
```
---
## 🎮 Controls
|
 Key 
|
 Action 
|
|
---
|
---
|
|
`←`
 / 
`A`
|
 Move counting line left 
|
|
`→`
 / 
`D`
|
 Move counting line right 
|
|
`R`
|
 Reset counters and save current reports 
|
|
`F`
|
 Toggle fullscreen 
|
|
`ESC`
|
 Exit and save final reports 
|
---
## ⚙️ Configuration
All settings are defined as constants at the top of [`counter.py`](counter.py):
|
 Constant 
|
 Default 
|
 Description 
|
|
---
|
---
|
---
|
|
`MODEL_PATH`
|
`"yolov8n.pt"`
|
 Path to the YOLO model weights 
|
|
`VIDEO_SOURCE`
|
`0`
|
 Webcam index or video file path 
|
|
`LINE_X_INIT`
|
`300`
|
 Initial X position of the counting line (pixels) 
|
|
`BUFFER`
|
`10`
|
 Dead-zone width on each side of the line (pixels) 
|
|
`LINE_SPEED`
|
`10`
|
 Line movement speed per key press (pixels) 
|
|
`CSV_PATH`
|
`"people_log.csv"`
|
 Output path for the CSV report 
|
|
`EXCEL_PATH`
|
`"people_report.xlsx"`
|
 Output path for the Excel report 
|
|
`XML_PATH`
|
`"people_log.xml"`
|
 Output path for the XML report 
|
---
## 📁 Project Structure
```
people-counter/
│
├── counter.py            # Main application — detection, tracking, UI, and reports
├── test_counter.py       # Unit tests (pytest)
├── yolov8n.pt            # YOLOv8 nano model weights
│
├── people_log.csv        # Generated: raw event log (created on run)
├── people_log.xml        # Generated: XML event log (created on run)
├── people_report.xlsx    # Generated: Excel report with pie chart (created on run)
│
└── README.md
```
---
## 🧪 Running Tests
The project includes a comprehensive test suite covering all core logic — no webcam or display required.
```bash
pip install pytest
pytest test_counter.py -v
```
### Test Coverage
|
 Module 
|
 Tests 
|
|
---
|
---
|
|
`get_side`
|
 Side determination, boundary conditions, zero buffer 
|
|
`detect_crossing`
|
 Entry/exit detection, slow walks, duplicates, multi-ID 
|
|
`prune_stale_tracks`
|
 Stale ID cleanup, empty states 
|
|
`handle_key`
|
 ESC, fullscreen toggle, arrow/WASD movement, clamping, reset 
|
|
`generate_reports`
|
 CSV/Excel/XML creation, empty data, summary cell values 
|
|
`generate_xml`
|
 XML structure, summary counts, event nodes 
|
|
`init_capture`
|
 Failure/success cases for camera/file opening 
|
|
`reset_counter`
|
 State clearing, report saving, no-crash on empty data 
|
---
## 📊 Output Reports
### CSV (`people_log.csv`)
```csv
TrackID,Type,Time
1,ENTRY,2026-05-27 10:00:01
2,ENTRY,2026-05-27 10:00:05
3,EXIT,2026-05-27 10:00:12
```
### XML (`people_log.xml`)
```xml
<?xml version='1.0' encoding='utf-8'?>
<PeopleLog generated="2026-05-27 10:01:00">
  <Summary>
    <TotalEntries>2</TotalEntries>
    <TotalExits>1</TotalExits>
  </Summary>
  <Events>
    <Event>
      <TrackID>1</TrackID>
      <Type>ENTRY</Type>
      <Time>2026-05-27 10:00:01</Time>
    </Event>
    ...
  </Events>
</PeopleLog>
```
### Excel (`people_report.xlsx`)
- Full event log on the main sheet.
- Summary cells with total entry and exit counts.
- Auto-generated **Entry vs. Exit pie chart**.
---
## 🏗️ Architecture Overview
```
run_counter()
│
├── init_capture()          — Open camera / video file
├── YOLO.track()            — Person detection + tracking (class 0 only)
├── detect_crossing()       — State-based crossing logic (LEFT/RIGHT committed side)
│   └── get_side()          — Classify point relative to line + buffer
├── prune_stale_tracks()    — Prevent unbounded memory growth
├── draw_*()                — HUD, bounding boxes, counting line
├── handle_key()            — Keyboard controls (ESC, R, F, arrows, WASD)
└── generate_reports()      — CSV + Excel + XML output
    └── generate_xml()
```
---
## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests.
4. Ensure all tests pass: `pytest test_counter.py -v`
5. Open a Pull Request.
---
## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
