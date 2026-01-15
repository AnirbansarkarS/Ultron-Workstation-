import cv2
from ui.widgets import Button, Panel

class UIManager:
    def __init__(self, width, height, tool_manager):
        self.width = width
        self.height = height
        self.tool_manager = tool_manager
        self.panels = []
        self.setup_ui()
        
        # Interaction state
        self.last_cursor_pos = (0, 0)
        self.is_cursor_active = False

    def setup_ui(self):
        # --- Tools Panel (Left) ---
        panel_w = 80
        panel_h = 250
        panel_x = 10
        # Bottom left: height - panel_h - padding
        panel_y = self.height - panel_h - 20
        
        tools_panel = Panel(panel_x, panel_y, panel_w, panel_h, "Tools")
        
        btn_h = 40
        btn_gap = 10
        # Start drawing buttons a bit down from panel top (title space)
        start_y = panel_y + 30
        
        def set_tool_cb(name):
            self.tool_manager.set_tool(name)

        tools = ["DRAW", "ERASE", "SELECT", "MOVE"]
        for i, t in enumerate(tools):
             # Button y is absolute in this system
             y = start_y + i * (btn_h + btn_gap)
             # Button x relative to panel x
             btn = Button(panel_x + 10, y, 60, btn_h, t, color=(80, 80, 120), callback=set_tool_cb, payload=t)
             tools_panel.add_child(btn)
        
        self.panels.append(tools_panel)
        
        # --- Actions Panel (Bottom) ---
        actions_panel = Panel(self.width // 2 - 100, self.height - 80, 200, 60, "Actions")
        
        def undo_cb(): self.tool_manager.undo()
        def redo_cb(): self.tool_manager.redo()
        
        btn_undo = Button(self.width // 2 - 90, self.height - 50, 80, 30, "UNDO", color=(100, 60, 60), callback=undo_cb)
        btn_redo = Button(self.width // 2 + 10, self.height - 50, 80, 30, "REDO", color=(60, 100, 60), callback=redo_cb)
        
        actions_panel.add_child(btn_undo)
        actions_panel.add_child(btn_redo)
        
        self.panels.append(actions_panel)

    def update(self, cursor_x, cursor_y, is_clicking):
        """
        Update UI state.
        Returns True if UI captured the input (hovering/clicking), False if input should pass to World.
        """
        self.last_cursor_pos = (cursor_x, cursor_y)
        self.is_cursor_active = True # Assume valid if called
        
        captured = False
        
        for panel in self.panels:
            panel.update(cursor_x, cursor_y)
            if panel.contains(cursor_x, cursor_y):
                captured = True
                if is_clicking:
                    panel.handle_click(cursor_x, cursor_y)
        
        return captured

    def draw(self, frame):
        for panel in self.panels:
            panel.draw(frame)
            
        # Draw Active Tool Overlay
        cv2.putText(frame, f"TOOL: {self.tool_manager.active_tool}", (self.width - 200, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
