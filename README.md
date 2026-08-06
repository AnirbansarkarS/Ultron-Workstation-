# Ultron Workstation 🖐️🧊
> **Vision-Driven Gesture-Controlled 3D Voxel Engine & Spatial Workstation**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-orange.svg)](https://google.github.io/mediapipe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Ultron Workstation** is a touchless, gesture-controlled 3D voxel editor and spatial visualization engine built in Python. By combining computer vision (`MediaPipe` hand skeleton tracking) with a custom CPU-based 3D software rendering engine (Model-View-Projection pipeline with depth sorting), Ultron Workstation allows users to build, manipulate, rotate, and interact with 3D voxel structures using natural hand gestures.

---

## 📸 Key Features

* 🎯 **3D Hand-Tracked Cursor**: Hover, aim, and position a floating 3D wireframe cursor directly between your fingers with real-time Z-depth spatial tracking.
* ✍️ **Gesture-Based Voxel Editing**:
  * **Draw / Place Voxels**: Point to position and place colored voxels into grid space.
  * **Erase Voxels**: Pinch or target near voxels with automatic proximity snapping to remove them.
  * **Color Palette Support**: Cycle through vibrant multi-color palettes on the fly.
* 🎥 **Intuitive Camera Controls**: Rotate camera angles dynamically around the 3D world using open palm movements.
* 📐 **Custom 3D Math Engine**: Built from scratch (`Vector3`, `Matrix4`, `Camera3D`) implementing perspective projection, view matrices, and transformation pipelines.
* 🎨 **Software Rendering Pipeline**: Depth sorting via Painter's Algorithm, Z-Buffer occlusion testing, frustum culling, and wireframe/cube face rendering using OpenCV.
* ⚡ **Gesture State Machine**: Multi-frame stability filtering to eliminate gesture flickering and ensure accidental-free interactions.
* 📊 **Spatial UI & Visual Feedback**: Integrated 3D/2D crosshair, coordinate axes indicator, depth color-bar visualization, and full-screen HUD display.

---

## 🛠️ Project Development Phases & Roadmap

Ultron Workstation was designed and developed across **5 distinct phases**, evolving from raw hand landmark tracking to a full spatial voxel CAD engine:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DEVELOPMENT ROADMAP                            │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│   Phase 1    │   Phase 2    │   Phase 3    │   Phase 4    │   Phase 5   │
│  Vision &    │ Coordinate   │  Pseudo-3D   │ Interactive  │ Object Manip│
│  Gestures    │   Mapping    │   Engine     │ Voxel Editor │    & HUD    │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
```

### Phase 1 — Computer Vision & Hand Landmark Tracking ✅
- Integrated **MediaPipe Hand Landmarker** to track 21 key skeleton points per hand in real-time.
- Engineered **Gesture Recognizer** for hand posture analysis (finger extension detection, joint angles, inter-finger distances).
- Implemented **Gesture State Machine** with multi-frame hysteresis filtering to ensure stable posture switching (Pointer, Pinch, Open Palm, Fist, Two-Hand Zoom).

### Phase 2 — Coordinate Mapping & Depth Estimation ✅
- Mapped 2D normalized camera coordinates $[0.0, 1.0]$ into continuous 3D world space $[-5, 5]$.
- Extracted Z-depth from hand skeleton landmarks and created a mapping pipeline converting hand depth $[-0.15, 0.05]$ into world depth $[0, 10]$.
- Added continuous depth bar UI with color-gradient feedback for Z-axis hand tracking.

### Phase 3 — Pseudo-3D Engine & MVP Graphics Pipeline ✅
- Built custom **Math3D Library**:
  - `Vector3`: Dot product, cross product, normalization, vector arithmetic.
  - `Matrix4`: Translation, scaling, rotation matrices, matrix multiplication.
- Developed virtual **`Camera3D`** with customizable Field of View (FOV) and View matrix calculation.
- Implemented full **Model-View-Projection (MVP)** 3D-to-2D screen rendering pipeline.
- Implemented depth-sorting algorithms (Painter's Algorithm & Z-Buffer simulation) for correct polygon occlusion and rendering order.

### Phase 4 — Real-Time Interactive Voxel Editor ✅
- Created **`VoxelEditor`** engine for managing voxel grid state, placement limits, and color palettes.
- Added grid-snapping spatial logic to convert hand movement into discrete voxel grid coordinates $(x, y, z)$.
- Integrated cooldown timers and proximity search (`find_nearest_voxel`) to prevent voxel spam and enable clean erasing.
- Implemented dynamic 3D cursor rendering with active mode coloring (Green for Draw, Red for Erase, Yellow for Camera Rotate).

### Phase 5 — Spatial Object Transformations & Advanced Control ✅
- Added world object transformation support (`Matrix4` scale, translation, rotation).
- Implemented two-handed gesture controls for zooming/scaling 3D object structures.
- Added dynamic coordinate frame axes $(X, Y, Z)$ rendering at object origin.
- Polished full-screen display HUD showing frame rate (FPS), mode status, voxel counts, and camera coordinates.

---

## 🎮 Gesture Control Guide

| Gesture | Hand Posture | Action / Mode | Description |
| :--- | :--- | :--- | :--- |
| ☝️ **Pointer / Gun** | Index finger & Thumb extended | `DRAW` | Aims floating cursor; places a new voxel at current grid location |
| 👌 **Pinch** | Index tip & Thumb tip together | `ERASE` | Snaps to nearest voxel near cursor and erases it |
| ✋ **Open Palm** | Flat open hand | `ROTATE_CAM` | Moving hand rotates camera around the 3D scene origin |
| ✊ **Fist** | Closed hand | `HOLD / FREEZE` | Pauses manipulation; holds current cursor & camera state |
| 🖐️🖐️ **Two Hands** | Both hands active | `SCALE / ZOOM` | Moving hands apart/together scales the 3D voxel structure |

---

## 📁 Repository Structure

```
Ultron-Workstation-/
├── main.py                      # Main entrypoint: camera loop, vision processing & render pipeline
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── PHASE3_README.md             # Technical documentation for Phase 3 3D Graphics Engine
│
├── vision/                      # Computer Vision & Hand Skeleton Tracking
│   ├── camera.py                # OpenCV webcam capture wrapper
│   ├── hand_tracker.py          # MediaPipe landmark inference engine
│   ├── depth_mapper.py          # Hand landmark Z-depth extraction & world mapping
│   ├── coordinate_space.py      # Screen-to-world spatial transformations
│   ├── landmark_utils.py        # Point normalization & math helpers
│   └── hand_landmarker.task     # MediaPipe ML model binary
│
├── gestures/                    # Gesture Recognition & State Filtering
│   ├── recognizer.py            # Geometric posture evaluation (Pointer, Pinch, Palm, etc.)
│   ├── state_machine.py         # Multi-frame hysteresis gesture state stabilizer
│   └── finger_state.py          # Finger extension state calculator
│
├── math3d/                      # Custom 3D Software Math Library
│   ├── vector.py                # Vector3 class with vector algebra
│   ├── matrix.py                # Matrix4 class with 4x4 matrix transformations
│   └── projection.py            # Perspective projection & view matrix calculations
│
├── render/                      # 3D Software Rendering Pipeline
│   ├── camera3d.py              # Virtual Camera3D class (FOV, position, rotation)
│   ├── pseudo3d.py              # World-to-Screen 3D projection pipeline
│   └── zbuffer.py               # Depth buffer array for software occlusion
│
├── world/                       # Voxel Data Structures & Editor Logic
│   ├── voxel_grid.py            # 3D Spatial Voxel storage & matrix transformations
│   ├── voxel_editor.py          # Voxel edit mode state machine, cooldowns & placement
│   └── voxel_ops.py             # Cube vertex generation & Painter's depth sorting
│
├── ui/                          # Workspace HUD & Spatial Overlay UI
│   ├── hud.py                   # FPS and status overlay text
│   ├── cursor.py                # 2D crosshair & 3D cursor rendering
│   └── panels.py                # Workspace panels & UI borders
│
├── tools/                       # Tool State & History Management
│   ├── color_picker.py          # Voxel color selection utilities
│   ├── history.py               # Undo / Redo action stack
│   └── tool_state.py            # Active tool configuration
│
└── experiments/                 # Test Suites & Prototypes
    ├── transform_test.py        # Matrix transformation tests
    ├── hand_depth_test.py       # Hand depth calibration script
    └── voxel_projection_test.py # MVP rendering pipeline test suite
```

## ⚡ Installation & Setup

### Prerequisites
* **Python 3.9+** installed on your system.
* A working **Webcam** for hand tracking.

### Step-by-Step Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AnirbansarkarS/Ultron-Workstation-.git
   cd Ultron-Workstation-
   ```

2. **Set Up Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Ultron Workstation**:
   ```bash
   python main.py
   ```

---

## 📐 Technical Deep-Dive

### 1. Spatial Coordinate Pipeline
Ultron Workstation converts raw 2D webcam coordinates and MediaPipe landmarks into 3D world grid units through a multi-stage transformation:

$$\text{Landmark }(x, y, z)_{\text{MediaPipe}} \longrightarrow \text{World }(X, Y, Z)_{\text{World}} \longrightarrow \text{View Space} \longrightarrow \text{Clip / Screen Space }(x_{px}, y_{px})$$

```
MediaPipe Hand Landmark (Normalized 0.0 - 1.0)
   ↓ (Hand-to-World mapping & Z-depth scaling)
Local Grid Space (x, y, z)
   ↓ (Model Matrix Transformation)
World Space (X, Y, Z)
   ↓ (View Matrix - Camera Position & Orientation)
Camera View Space
   ↓ (Perspective Projection Matrix)
Homogeneous Clip Space (x, y, z, w)
   ↓ (Perspective Divide by w)
Normalized Device Coordinates (NDC [-1, 1])
   ↓ (Viewport Transform)
Screen Coordinates (Pixels X, Y + Z-depth sorting)
```

### 2. Software Painter's Depth Algorithm
Since rendering is executed via CPU polygon drawing with OpenCV, depth occlusion is achieved using a distance-sorted **Painter's Algorithm**:
1. Voxel centers $(X, Y, Z)$ are transformed by the current model matrix.
2. Euclidean distance squared $d^2 = (X - C_x)^2 + (Y - C_y)^2 + (Z - C_z)^2$ relative to camera position $C$ is computed for all active voxels.
3. Voxels are sorted in **descending order** (farthest voxels rendered first, nearest voxels drawn last).
4. Vertices are projected onto the screen canvas with shaded faces and wireframe highlights.

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome! Feel free to open an Issue or submit a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
