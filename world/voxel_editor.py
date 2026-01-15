import numpy as np
import time
import math

class VoxelEditor:
    def __init__(self, voxel_grid, camera):
        """
        Initialize voxel editor.
        """
        from tools.tool_manager import ToolManager
        self.voxel_grid = voxel_grid
        self.camera = camera
        self.tool_manager = ToolManager(voxel_grid)
        
        self.cursor_pos = (0, 0, 0)
        
        # Camera Rotate control
        self.rotation_base = None
        self.rotation_start = (0, 0, 0)
        
        # Object Manipulation Control
        self.manip_start_pos = None
        self.manip_start_scale = 1.0
        self.manip_start_rotation = None
        self.manip_initial_centroids = None
        self.manip_initial_dist = 0.0
    
    @property
    def mode(self):
        return self.tool_manager.active_tool

    def hand_to_world(self, hand_x, hand_y, hand_z, screen_width, screen_height):
        """
        Convert hand position to 3D world coordinates.
        
        Args:
            hand_x, hand_y: Normalized hand position [0, 1]
            hand_z: Hand depth (MediaPipe Z-value)
            screen_width, screen_height: Screen dimensions
        
        Returns:
            (x, y, z) in world space (snapped to grid)
        """
        from vision.depth_mapper import map_depth_to_world
        
        # Map hand X/Y to world X/Y (centered, scaled)
        # Hand [0, 1] → World [-5, 5] for a 10-unit workspace
        world_x = (hand_x - 0.5) * 10
        world_y = -(hand_y - 0.5) * 10  # Flip Y (hand Y goes down)
        
        # Map hand Z to world Z
        world_z = map_depth_to_world(hand_z, min_depth=-3, max_depth=3)
        
        # Snap to integer grid
        grid_x = round(world_x)
        grid_y = round(world_y)
        grid_z = round(world_z)
        
        # Apply inverse object transform to get local grid coordinates
        raw_point = (world_x, world_y, world_z)
        
        if self.voxel_grid.transform:
            inv = self.voxel_grid.transform.inverse()
            if inv:
                lx, ly, lz, _ = inv.transform_point(raw_point)
                return (round(lx), round(ly), round(lz))
        
        return (grid_x, grid_y, grid_z)

    def update_mode(self, gesture):
        """
        Update tool state based on gesture.
        Handle transient modes (Rotate, Hold).
        """
        current_tool = self.tool_manager.active_tool
        
        # Transient Modes
        if gesture == "open_palm":
            if current_tool != "ROTATE_CAM":
                self.previous_tool = current_tool
                self.tool_manager.set_tool("ROTATE_CAM")
        elif gesture == "fist":
            if current_tool != "HOLD":
                self.previous_tool = current_tool
                self.tool_manager.set_tool("HOLD")
        else:
            # Revert if we were in a transient mode and gesture ended
            if current_tool == "ROTATE_CAM" or current_tool == "HOLD":
                if hasattr(self, 'previous_tool'):
                    self.tool_manager.set_tool(self.previous_tool)
                else:
                    self.tool_manager.set_tool("DRAW")
        
    def use_current_tool(self, gesture):
        """
        Execute the current tool's action if the gesture matches.
        """
        if self.mode == "ROTATE_CAM":
            return # Handled separately
            
        # Map gestures to "Action Active"
        is_active = False
        if self.mode == "DRAW" and (gesture == "pointer" or gesture == "index_point"):
            is_active = True
        elif self.mode == "ERASE" and gesture == "pinch":
            is_active = True
        elif self.mode == "COLOR_PICK" and gesture == "pinch":
             is_active = True
             
        return self.tool_manager.use_tool(self.cursor_pos, is_active)

    def cycle_color(self):
        self.tool_manager.cycle_color()
    
    def get_current_color(self):
        return self.tool_manager.get_current_color()
    
    def start_rotation(self, hand_x, hand_y):
        """
        Start camera rotation control.
        
        Args:
            hand_x, hand_y: Normalized hand position
        """
        self.rotation_base = (hand_x, hand_y)
        self.rotation_start = self.camera.rotation
    
    def update_rotation(self, hand_x, hand_y):
        """
        Update camera rotation based on hand movement.
        
        Args:
            hand_x, hand_y: Current normalized hand position
        """
        if self.rotation_base is None:
            self.start_rotation(hand_x, hand_y)
            return
        
        # Calculate hand movement delta
        base_x, base_y = self.rotation_base
        delta_x = (hand_x - base_x) * 3.0  # Scale for sensitivity
        delta_y = (hand_y - base_y) * 3.0
        
        # Apply rotation
        rx, ry, rz = self.rotation_start
        new_rotation = (rx + delta_y, ry + delta_x, rz)
        self.camera.rotation = new_rotation
    
    
    def update_manipulation(self, landmarks_list, screen_w, screen_h):
        """
        Handle object manipulation based on mode.
        """
        if self.mode == "GRAB":
            # Use Index tip of first hand
            if not landmarks_list: return
            landmarks = landmarks_list[0]
            ind = landmarks[8] # Index tip
            wrist = landmarks[0]
            mid_mcp = landmarks[9]
            
            # --- TRANSLATION ---
            # Convert to world scale (approx)
            wx = (ind[0] - 0.5) * 10
            wy = -(ind[1] - 0.5) * 10
            # Use constant depth for now or map hand Z
            from vision.depth_mapper import map_depth_to_world
            wz = map_depth_to_world(ind[2], min_depth=-3, max_depth=3)
            
            current_pos = np.array([wx, wy, wz])
            
            # --- ROTATION (Orientation) ---
            # Vector from Wrist to Middle MCP (Hand direction)
            v_current = np.array([mid_mcp[0] - wrist[0], -(mid_mcp[1] - wrist[1]), mid_mcp[2] - wrist[2]])
            # Normalize
            norm = np.linalg.norm(v_current)
            if norm > 0: v_current /= norm
            else: v_current = np.array([0, 1, 0])

            if self.manip_start_pos is None:
                self.manip_start_pos = current_pos
                self.manip_start_rotation = v_current
                return
            
            # Apply Translation Delta
            delta = current_pos - self.manip_start_pos
            
            # --- AXIS LOCKING (Soft Snap) ---
            # If movement is dominant on one axis, lock to it
            dx, dy, dz = delta
            adx, ady, adz = abs(dx), abs(dy), abs(dz)
            
            # Threshold for locking (must be moving somewhat to lock)
            if max(adx, ady, adz) > 0.5: 
                # Check dominance (e.g. 2x larger than others)
                if adx > 2.0 * ady and adx > 2.0 * adz:
                    delta[1] = 0; delta[2] = 0 # Lock to X
                elif ady > 2.0 * adx and ady > 2.0 * adz:
                    delta[0] = 0; delta[2] = 0 # Lock to Y
                # Z locking omitted for now
            
            self.voxel_grid.translate(delta[0], delta[1], delta[2])
            
            # Apply Rotation Delta
            if self.manip_start_rotation is not None:
                # Calculate rotation from start_vector to current_vector
                axis = np.cross(self.manip_start_rotation, v_current)
                dot = np.dot(self.manip_start_rotation, v_current)
                dot = max(-1.0, min(1.0, dot)) # Clamp
                angle = math.acos(dot)
                
                # Threshold to avoid jitter
                if abs(angle) > 0.02: # ~1 degree
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 0.001:
                        axis /= axis_norm
                        # Simplified Rotation Mapping
                        if abs(axis[0]) > abs(axis[1]) and abs(axis[0]) > abs(axis[2]):
                            self.voxel_grid.rotate('x', angle * np.sign(axis[0]))
                        elif abs(axis[1]) > abs(axis[2]):
                            self.voxel_grid.rotate('y', angle * np.sign(axis[1]))
                        else:
                            self.voxel_grid.rotate('z', angle * np.sign(axis[2]))
                            
                            # Note: To prevent continuous spinning if we don't update reference,
                            # we update the reference vector here.
                            self.manip_start_rotation = v_current

            # Update start pos
            self.manip_start_pos = current_pos
            
        elif self.mode == "SCALE":
            if len(landmarks_list) < 2: return
            
            # Distance between wrists or index tips
            p1 = np.array(landmarks_list[0][0]) # Wrist 1
            p2 = np.array(landmarks_list[1][0]) # Wrist 2
            
            dist = np.linalg.norm(p1 - p2)
            
            if self.manip_initial_dist == 0.0:
                self.manip_initial_dist = dist
                return
            
            # Scaling factor
            if dist > 0:
                scale_delta = dist / self.manip_initial_dist
                # Dampen
                # scale_factor = 1.0 + (scale_delta - 1.0) * 0.1
                # But matrix multiplication accumulates.
                # If we want 1:1 scaling mapping:
                # We need to apply relative scale.
                
                # Option A: Apply small incremental scale.
                # If dist > initial, scale > 1.
                # We need to reset initial dist every frame? No, that drifts.
                # Better: compare to previous frame's dist?
                pass
            
            # Incremental approach
            if self.manip_initial_dist > 0:
                ratio = dist / self.manip_initial_dist
                # Apply if significant change
                if abs(ratio - 1.0) > 0.01:
                    self.voxel_grid.scale(ratio)
                    # Reset baseline to avoid exponential growth
                    self.manip_initial_dist = dist

    def reset_rotation(self):
        """Reset rotation control."""
        self.rotation_base = None