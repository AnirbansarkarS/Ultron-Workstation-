# Ultron Workstation - Hand-Controlled Voxel Editor

A gesture-controlled 3D voxel editor that uses computer vision (OpenCV + MediaPipe) to let you build and manipulate 3D structures with your hands.

## 🚀 Features

*   **Stable Workspace**: A precise, non-rotating 3D environment for content creation.
*   **Intuitive Cursor**: The 3D cursor floats exactly at the midpoint between your thumb and index finger, giving you a tactile sense of "holding" the voxels.
*   **Gesture Control Engine**:
    *   👌 **Pinch (Index + Thumb)**: **Draw / Place** a voxel.
    *   ✊ **Fist**: **Erase** voxels near the cursor.
    *   ✋ **Open Palm**: **Rotate** the camera view by moving your hand.
    *   ☝️ **Point**: **Aim** (cursor follows hand without drawing).
*   **Real-time Rendering**: Custom pseudo-3D engine with perspective projection, depth sorting, and occlusion.

## 🛠️ Installation

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 How to Run

```bash
python main.py
```

## 💡 Usage Guide

1.  **Start the App**: The camera feed will open with a 3D grid overlay.
2.  **Aim**: Hold your hand up. A "Ghost Voxel" (preview) will appear between your thumb and index finger.
3.  **Draw**: Pinch your thumb and index finger together to place a permanent voxel at that spot.
4.  **Erase**: Make a fist to erase any voxels near the cursor.
5.  **Rotate View**: Open your palm flat. As you move your hand, the camera rotates around the center of the world.
6.  **Exit**: Press `Esc` to quit.

## 🏗️ Technical Architecture

*   **Vision**: `mediapipe` for hand skeleton tracking (21 landmarks).
*   **Math**: Custom `Vector3` and `Matrix4` classes implementing a full Model-View-Projection (MVP) pipeline.
*   **Rendering**: CPU-based rendering using OpenCV (`cv2` lines and polygons) with Painter's Algorithm for depth sorting.
*   **Interaction**: State-machine based gesture recognition for stable input.
