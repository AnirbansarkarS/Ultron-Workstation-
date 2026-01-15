import cv2

class UIElement:
    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.is_hovered = False

    def contains(self, x, y):
        return (self.x <= x <= self.x + self.w) and (self.y <= y <= self.y + self.h)

    def update(self, cursor_x, cursor_y):
        self.is_hovered = self.contains(cursor_x, cursor_y)

    def draw(self, frame):
        pass

class Button(UIElement):
    def __init__(self, x, y, w, h, text, color=(100, 100, 100), callback=None, payload=None):
        super().__init__(x, y, w, h)
        self.text = text
        self.base_color = color
        self.hover_color = (min(color[0]+50, 255), min(color[1]+50, 255), min(color[2]+50, 255))
        self.callback = callback
        self.payload = payload  # Value to pass to callback (e.g., tool name)

    def draw(self, frame):
        color = self.hover_color if self.is_hovered else self.base_color
        
        # Draw background
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 1)
        
        # Draw Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(self.text, font, scale, thickness)
        
        text_x = self.x + (self.w - text_w) // 2
        text_y = self.y + (self.h + text_h) // 2
        
        cv2.putText(frame, self.text, (text_x, text_y), font, scale, (255, 255, 255), thickness)

    def on_click(self):
        if self.callback:
            if self.payload:
                self.callback(self.payload)
            else:
                self.callback()
            return True
        return False

class Panel(UIElement):
    def __init__(self, x, y, w, h, title=""):
        super().__init__(x, y, w, h)
        self.title = title
        self.children = []
    
    def add_child(self, element):
        # Adjust child coordinates to be relative to panel? 
        # For simplicity, let's keep absolute coords or handle layouting manually.
        # Here we assume element has absolute coords.
        self.children.append(element)

    def update(self, cursor_x, cursor_y):
        super().update(cursor_x, cursor_y)
        for child in self.children:
            child.update(cursor_x, cursor_y)
    
    def draw(self, frame):
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Border
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (100, 100, 100), 1)
        
        if self.title:
             cv2.putText(frame, self.title, (self.x + 5, self.y + 20), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for child in self.children:
            child.draw(frame)

    def handle_click(self, x, y):
        """Returns True if a child handled the click."""
        if not self.contains(x, y):
            return False
            
        for child in self.children:
            if child.contains(x, y):
                if isinstance(child, Button):
                    return child.on_click()
                elif isinstance(child, Panel):
                     if child.handle_click(x, y):
                         return True
        return True # Clicked panel but no button (consumed)
