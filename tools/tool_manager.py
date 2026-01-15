import time
from abc import ABC, abstractmethod

class ToolAction:
    """Represents a reversible action."""
    def __init__(self, type, data, inverse_data):
        self.type = type
        self.data = data
        self.inverse_data = inverse_data

class ToolManager:
    def __init__(self, voxel_grid):
        self.voxel_grid = voxel_grid
        self.active_tool = "DRAW"  # DRAW, ERASE, SELECT, COLOR_PICK, MOVE_CAM
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50
        
        # Tool state
        self.selected_color_index = 0
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        # Interaction state
        self.is_interacting = False
        self.last_action_time = 0
        self.action_cooldown = 0.15

    def set_tool(self, tool_name):
        if tool_name != self.active_tool:
            self.active_tool = tool_name
            print(f"Tool changed to: {tool_name}")

    def get_current_color(self):
        return self.colors[self.selected_color_index]

    def cycle_color(self):
        self.selected_color_index = (self.selected_color_index + 1) % len(self.colors)
        print(f"Color changed to: {self.get_current_color()}")

    def set_color(self, index):
        if 0 <= index < len(self.colors):
            self.selected_color_index = index

    def push_action(self, action):
        """Record an action for undo."""
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()  # Clear redo on new action

    def undo(self):
        if not self.undo_stack:
            return False
            
        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        
        if action.type == "PLACE_VOXEL":
            pos = action.data['pos']
            # Inverse of place is delete
            if pos in self.voxel_grid.grid:
                del self.voxel_grid.grid[pos]
            return True
            
        elif action.type == "ERASE_VOXEL":
            pos = action.inverse_data['pos']
            color = action.inverse_data['color']
            # Inverse of erase is place
            self.voxel_grid.set_voxel(pos, color)
            return True
            
        return False

    def redo(self):
        if not self.redo_stack:
            return False
            
        action = self.redo_stack.pop()
        self.undo_stack.append(action)
        
        if action.type == "PLACE_VOXEL":
            pos = action.data['pos']
            color = action.data['color']
            self.voxel_grid.set_voxel(pos, color)
            return True
            
        elif action.type == "ERASE_VOXEL":
            pos = action.data['pos']
            if pos in self.voxel_grid.grid:
                del self.voxel_grid.grid[pos]
            return True
            
        return False

    def use_tool(self, pos, gesture_is_active):
        """
        Apply current tool at position.
        Returns result string or None.
        """
        now = time.time()
        if now - self.last_action_time < self.action_cooldown:
            return None

        if not gesture_is_active:
            self.is_interacting = False
            return None
            
        self.is_interacting = True
        
        if self.active_tool == "DRAW":
            if self.voxel_grid.get_voxel(pos) is None:
                color = self.get_current_color()
                self.voxel_grid.set_voxel(pos, color)
                
                # Record Action
                action = ToolAction("PLACE_VOXEL", 
                                  data={'pos': pos, 'color': color},
                                  inverse_data={'pos': pos})
                self.push_action(action)
                
                self.last_action_time = now
                return "VOXEL_PLACED"
                
        elif self.active_tool == "ERASE":
            existing_color = self.voxel_grid.get_voxel(pos)
            if existing_color is not None:
                if pos in self.voxel_grid.grid:
                    del self.voxel_grid.grid[pos]
                    
                    # Record Action
                    action = ToolAction("ERASE_VOXEL",
                                      data={'pos': pos},
                                      inverse_data={'pos': pos, 'color': existing_color})
                    self.push_action(action)
                    
                    self.last_action_time = now
                    return "VOXEL_ERASED"
        
        elif self.active_tool == "COLOR_PICK":
             existing_color = self.voxel_grid.get_voxel(pos)
             if existing_color is not None:
                 try:
                     idx = self.colors.index(existing_color)
                     self.selected_color_index = idx
                     self.active_tool = "DRAW" # Auto-switch back to draw
                     return "COLOR_PICKED"
                 except ValueError:
                     pass

        return None
