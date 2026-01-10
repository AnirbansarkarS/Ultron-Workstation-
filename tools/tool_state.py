"""
Draw / erase / select state management.
"""

class ToolState:
    DRAW = 0
    ERASE = 1
    SELECT = 2

    def __init__(self):
        self.current_tool = self.DRAW
