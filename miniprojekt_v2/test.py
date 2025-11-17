import cv2
from pathlib import Path
import json

image_dir_path = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\val\images")
image_data = Path(r"C:\Users\Daniel K\Desktop\data_splits_ready\val\val.json")

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

def get_image_path(image_index, image_dir=image_dir_path):
    return sorted(image_dir.glob("*"))[image_index]

def get_data(data_path=image_data):
    with open(data_path) as f:
        return json.load(f)


def show_first_image(image_dir, index=0):
    first_image = sorted(image_dir.glob("*"))[index]
    print(first_image)
    w, h, c = cv2.imread(str(first_image)).shape
    print(w)
    print(h)
    print(c)
    
    first_image = cv2.imread(first_image)
    cv2.imshow("Runescape image 1", first_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_image_w_h(image_path):
    h, w, _ = cv2.imread(str(image_path)).shape
    return h, w


def file_name(image_dir, index):
    first_image = sorted(image_dir.glob("*"))[index]
    filename = first_image.name
    return filename

def get_image_data(image_index, data=get_data()):

    for i in data:
        if file_name(image_dir_path, image_index) in i["image"]:
            return i
    print("Error: couldnt find image data")
            

def draw(image, x1, y1, bb_top_right, bb_lower_left, class_name, conf=1):
        rect_thickness = 2
        cv2.rectangle( #cv2.rectangle(image, start_point, end_point, color, thickness)
            image, 
            bb_top_right, 
            bb_lower_left, 
            CLASS_COLORS[class_name],
            rect_thickness
                        )
        cv2.putText(
            image,
            f"{class_name}, {conf}",
            (x1, y1 - 5),             # Text location 
            cv2.FONT_HERSHEY_SIMPLEX,           # Font
            1,                                  # Font scale
            CLASS_COLORS[class_name], # Color
            rect_thickness                      # thickness
        )



def show_first_bb(image_index):
    image_path=get_image_path(image_index)
    h, w = get_image_w_h(image_path)
    image_data = get_image_data(image_index)
    image__label_data_first_box=image_data["label"][0]


    bw = int(image__label_data_first_box["width"] / 100*w)
    bh = int(image__label_data_first_box["height"] / 100*h)
    
    
    x1 = int(image__label_data_first_box["x"] / 100*w)
    y1 = int(image__label_data_first_box["y"] / 100*h)
    x2 = x1 + bw
    y2 = y1 + bh

    img = cv2.imread(image_path)
    draw(img, x1, y1, (x1,y1), (x2, y2), image__label_data_first_box["rectanglelabels"][0])
    
    cv2.imshow("Runescape image 1", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def show_all_bb(image_index):
    image_path=get_image_path(image_index)
    h, w = get_image_w_h(image_path)
    image_data = get_image_data(image_index)
    img = cv2.imread(image_path)

    for bb in image_data["label"]: # bb = Bounding Box
        bw = int(bb["width"] / 100*w)
        bh = int(bb["height"] / 100*h)
    
    
        x1 = int(bb["x"] / 100*w)
        y1 = int(bb["y"] / 100*h)
        x2 = x1 + bw
        y2 = y1 + bh

    
        draw(img, x1, y1, (x1,y1), (x2, y2), bb["rectanglelabels"][0])
    
    cv2.imshow(f"Runescape image {image_index}", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



show_all_bb(0)


#print(get_image_data(0))







#file_name()
#print(data[0]["image"])
#print(data[0]["id"])
#print(data[0]["label"][0])


#show_first_image()