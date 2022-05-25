import cv2
import pandas as pd 
import numpy as np 

image_path = "../input/"
box_path="../input/"

figures_path = "../figures/"

def visualize_textboxes(img_name, save_img = False):
    """returns invoices images with textboxes"""
    image = image_path + img_name + '.jpg'
    box = box_path + img_name + '.csv'
    img = cv2.imread(image)
    with open(box) as topo_file:
        for line in topo_file:
            coor = line.split(',')
            #print(coor)
            x1,y1,x3,y3 = int(coor[0]),int(coor[1]),int(coor[2]),int(coor[3])
            text = coor[4].strip('\n').strip('\'')
            #print(x1,y1,x3,y3,text)

            img = cv2.rectangle(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
            img = cv2.putText(img, text, (x1, y1 - 1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1)

    if save_img:
        cv2.imwrite(figures_path + img_name + '_withtext.jpg', img)

def visualize_labels(img_name, save_img = False):
    """returns invoices with manually annontated labels"""
    image = image_path + img_name + '.jpg'
    box_labels = box_path + img_name + '.csv'
    df = pd.read_csv(box_labels)
    df.dropna(inplace=True)
    img = cv2.imread(image)

    for index, rows in df.iterrows():                   
            text = rows[4].upper() 

            x1,y1,x3,y3 = rows[0],rows[1],rows[2],rows[3]

            img = cv2.rectangle(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
            img = cv2.putText(img, text, (x1, y1 - 1),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

    if save_img:
        cv2.imwrite(figures_path + img_name + '_withlabels.jpg', img)

if __name__ == "__main__":
    visualize_textboxes('1', save_img = True)
    visualize_labels('1', save_img = True)