import cv2 

class vizualiser(self, image, modelOutput):

    CLASS_COLORS = { 
        "Adamant":        (80, 120, 70),    # dull green
        "Clay":           (200, 170, 120),  # pale brown/beige
        "Coal":           (30, 30, 30),     # black
        "Copper":         (170,100,50),     # orange/brown 
        "Gold":           (212,175,55),     # gold
        "Iron":           (130,120,110),    # gray with tint of brown
        "Mined":          (120,120,120),    # dark grey
        "Mithril":        (110,150,200),    # pale blue
        "Motherload_ore": (150,120,60),     # golden brown
        "Removable_ore":  (140,140,140),    # light grey
        "Runeite":        (45, 75, 160),    # deep blue
        "Silver":         (200,200,210),    # light silver
        "Tin":            (170,170,150),    # grey with yellow tint
    }

    def __init__(self):
        
        self.image = image
        for item in modelOutput:
            self.x1, self.y1, self.x2, self.y2 = [item["x1"], item["y1"], item["x2"], item[y2]] # Skal placere koordinaterne til bounding boksens hjørner i variablerne.

        self.bb_top_right = (self.x1, self.y1)
        self.bb_lower_left = (self.x2, self.y2)

        self.class_id = []        
        self.class_name = [] 
        self.conf = []

    def draw(self)
        rect_thickness = 2
        cv2.rectangle( #cv2.rectangle(image, start_point, end_point, color, thickness)
            self.image, 
            self.bb_top_right, 
            self.bb_lower_left, 
            self.CLASS_COLORS[self.class_name],
            rect_thickness
                        )
        cv2.putText(
            self.image,
            f"{self.class_name}, {self.conf}",
            (self.x1, self.y1 - 5),             # Text location 
            cv2.FONT_HERSHEY_SIMPLEX,           # Font
            1,                                  # Font scale
            self.CLASS_COLORS[self.class_name], # Color
            rect_thickness                      # thickness
        )
