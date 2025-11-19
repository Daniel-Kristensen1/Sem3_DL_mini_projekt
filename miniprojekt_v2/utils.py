
import torch
import numpy as np
import config
import torchvision.transforms as T
from PIL import Image
import cv2
from pathlib import Path
import json

image_dir_path = config.TEST_IMAGES
image_data = config.TEST_JSON


#####################################
########## Path Handeling ###########
#####################################

def get_image_path(image_dir_path, image_index):
    """
    Henter stien til billedet ved et givent index i en mappe med billeder.

    Parametre:
    image_dir_path (path): Stien til mappen med billeder.
    image_index (int): Billedets index i mappen.

    Returnerer:
    Path: Stien til billedet ved indexet.
    """  

    return sorted(image_dir_path.glob("*"))[image_index]


#####################################
########## Data Handeling ###########
#####################################

def get_data(data_path):
    """
    Henter data fra en json-fil af annoteret data.   

    Parametre:
    data_path (path): Stien til json filen med annoteret billede data. 

    Returnerer:
    List: En liste med dictionaries som indeholder annoteret billededata.
    """  

    with open(data_path) as data:
        return json.load(data)

def get_image_data(image_dir_path, image_index, data):
    """
    Henter det den annoterede data fra json filen, 
    som tilhøre et specfiikt billede.

    Parametre:
    image_dir_path (path): Stien til mappen med billeder.
    image_index (int): Billedets index i mappen.
    data (list): Liste med alt annoteret data fra billederne i mappen. 

    Returnerer:
    dict: En dictionary med den respektive data til det specifikke billede i mappen
    """ 

    for i in data:
        if file_name(image_dir_path, image_index) in i["image"]:
            return i
    print("Error: couldnt find image data")

def get_class_info(id):
    """
    Henter information om klassens navn og bounding boks farve. 

    Parametre:
    id (int): klassen id / index i variablen config.CLASSES

    Returnerer:
    str: navnet på klassen
    tuple: RGB farver 
    """ 
    class_name = config.CLASSES[id]
    class_color = config.CLASS_COLORS[class_name]
    return class_name, class_color      

#####################################
########## Image Handeling #########
#####################################

def file_name(image_dir_path, index):
    """
    Henter navnet på billedet for at kunne sammenligne billede navnet 
    med det respektive dictionary i json filen, som indeholder 
    den annoterede data til præcis dette billede

    Parametre:
    image_dir_path (path): Stien til mappen med billeder.
    image_index (int): Billedets index i mappen.

    Returnerer:
    str: fil-navnet til billedet
    """ 

    first_image = get_image_path(image_dir_path, index)
    filename = first_image.name
    return filename

def get_image_w_h(image_path):
    """
    Henter 'height' og 'width' fra det originale billede, for at kunne skalere bounding boksenes x og y koordinater. 

    Parametre:
    image_path (path): Stien til et enkelt billede

    Returnerer:
    int: højden(height) og breden(width) af billedet
    """  

    h, w, _ = cv2.imread(str(image_path)).shape
    return h, w


def image_to_tensor(path):
    img = Image.open(path).convert("RGB")

    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor()
    ])

    img_tensor = transform(img)#.unsqueeze(0)   # Add batch dimension
    img_tensor = img_tensor.to(config.DEVICE)
    return [img_tensor]


#####################################
########## DRAW BOUNDING BOX ########
#####################################

def draw(image, x1, y1, bb_top_right, bb_lower_left, class_name, conf=1):
    """
    Tegner bounding bokse på input billedet. 
    Bokse er farvelagt og navngivet efter klasse.
    Tilføger også en konfidence score til boksen.

    Parametre:
    image ():
    x1 ():
    x2 ():
    bb_top_right ():
    bb_lower_left ():
    class_name ():
    conf (): 
    
    """ 
    rect_thickness = 2
    cv2.rectangle( #cv2.rectangle(image, start_point, end_point, color, thickness)
        image, 
        bb_top_right, 
        bb_lower_left, 
        config.CLASS_COLORS[class_name],
        rect_thickness
                    )
    cv2.putText(
        image,
        f"{class_name}, {conf}",
        (x1, y1 - 5),             # Text location 
        cv2.FONT_HERSHEY_SIMPLEX,           # Font
        1,                                  # Font scale
        config.CLASS_COLORS[class_name], # Color
        rect_thickness                      # thickness
    )


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


def show_all_bb_inf(image_path, image_pred_boxes, image_pred_labels, image_scores):
    h, w = get_image_w_h(image_path)
    img = cv2.imread(image_path)

    for index, bb in enumerate(image_pred_boxes): # bb = Bounding Box
        x1, y1, x2, y2 = bb.int().tolist()
        
        x1 = int(x1 / 640*w)
        y1 = int(y1 / 640*h)
        x2 = int(x2 /  640*w)
        y2 = int(y2 / 640*h)

        class_id = image_pred_labels[index]
        class_name = config.CLASSES[class_id-1]
        draw(img, x1, y1, (x1,y1), (x2, y2), class_name=class_name, conf=image_scores[index-1])
    
    print(" Bounding boxes drawn - Image shown on screen" )
    cv2.imshow(f"Runescape image - Inference", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(" Window closed.")

#show_all_bb(0)


#print(get_image_data(0))







#file_name()
#print(data[0]["image"])
#print(data[0]["id"])
#print(data[0]["label"][0])


#show_first_image()

