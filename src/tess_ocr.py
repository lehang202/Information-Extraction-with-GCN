import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files/Tesseract-OCR/tesseract.exe'

image = "../input/1.jpg"
output ="../figures/tess_1.jpg"

img = cv2.imread(image)

custom_config = r'--oem 3 --psm 6'

d = pytesseract.image_to_data(img, output_type=Output.DICT) #, config=custom_config)

#for graph modeling
xmin,ymin,xmax,ymax,Object = [],[],[],[],[]
df = pd.DataFrame()

n_boxes = len(d['text'])

for i in range(n_boxes):
    if int(float(d['conf'][i])) >= 0.74:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #print(x,y,w,h)
        text = d['text'][i]

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        img = cv2.putText(img, text, (x, y - 1),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)


        img_height, img_width =  img.shape[0], img.shape[1]

        xmin.append(x)
        ymin.append(y)
        xmax.append(x + w)
        ymax.append(y + h)
        Object.append(text)

df['xmin'], df['ymin'], df['xmax'], df['ymax'], df['Object']  = xmin,ymin,xmax,ymax,Object
#df['0'], df['1'], df['2'], df['3'], df['4']  = xmin,ymin,xmax,ymax,Object
df = df[df.Object != " "]

print(df)

df.to_csv('scratchpart' + '.csv' ,index = False)
cv2.imwrite('scratchpart' + '.jpg', img)

cv2.imwrite(output, img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()