import cv2
import numpy as np
import sys
import os
import glob
from skimage.measure import label, regionprops

USERIMAGEPATH = '/Users/henrypulley/Desktop/Umbrella/Academics and Career/Academics/Senior Project/Senior Project II/GCSF-keras-yolo3/images'

def show_image(img):
    cv2.imshow('t',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_dir = USERIMAGEPATH
annotation_file = sys.argv[1]

rgb_image_paths = glob.glob(image_dir + '/rgb*')
img_numbers = [(x.split("_")[-1]).split(".")[0] for x in rgb_image_paths]
img_numbers.sort(key=int)

''' SETUP ANNOTATION FILE FOR TRAINING DATA LIKE SO:

image_file_path box1 box2 ... boxN
where each box is -- x_min,y_min,x_max,y_max,class_id
-----------
path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
path/to/img2.jpg 120,300,250,600,2
...
...
...
'''


for img_number in img_numbers:
    rgb_image_path = image_dir + "/rgb_" + img_number + ".png"
    seg_image_path = image_dir + "/seg_" + img_number + ".png"

    annotation_line = os.path.abspath(rgb_image_path) + " 0 "

    seg_img = cv2.imread(seg_image_path, cv2.IMREAD_GRAYSCALE)

    seg_img[seg_img == 114] = 0 # Set landscape to 0
    seg_img[seg_img == 126] = 1 # Crater1
    seg_img[seg_img == 168] = 1 # Crater2
    seg_img[seg_img == 121] = 2 # sand
    seg_img[seg_img == 129] = 3 # mountain
    seg_img[seg_img == 130] = 4 # rock

    crater_idxs = np.where((seg_img == 1))
    sand_idxs = np.where((seg_img == 2))
    mountain_idxs = np.where((seg_img == 3))
    rock_idxs = np.where((seg_img == 4))


    terrain_idxs = {
        'crater': crater_idxs,
        'sand': sand_idxs,
        'mountain': mountain_idxs,
        'rock': rock_idxs
    }

    terrain_bbox_thresholds = {
        'crater': 100,
        'sand': 100,
        'mountain': 100,
        'rock': 100
    }

    terrain_class_ids = {
        'crater': 0,
        'sand': 1,
        'mountain': 2,
        'rock': 3
    }

    # # BBox each piece of terrain in the image
    ol = annotation_line
    for terrain in list(terrain_idxs.keys()):
        class_id = terrain_class_ids[terrain]
        idxs = terrain_idxs[terrain]
        if len(idxs[0]):

            # Binary mask terrain
            seg_img[:] = 0
            seg_img[idxs] = 255

            label_img = label(seg_img, connectivity=1)
            props = regionprops(label_img)

            for prop in props:
                if prop.area >= terrain_bbox_thresholds[terrain]:
                    bbox = prop.bbox
                    annotation_line += "{},{},{},{},{} ".format(bbox[1], bbox[0], bbox[3], bbox[2], class_id)

    # If there were bboxs detected at all, write to annotation file
    if ol != annotation_line:
        with open(annotation_file, 'a') as f:
            f.write(annotation_line + "\n")