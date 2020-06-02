from keras.models import model_from_json
import numpy as np
import cv2
import argparse
import os
import ffmpeg
from openpyxl import  Workbook
from openpyxl.drawing.image import Image


class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Frustrated", "Disagree", "Tense", "Happy", "Frustrated", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(
                loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


#code to deal with arguments passed through cli.

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("fps")
args = parser.parse_args()

if args.source != 'webcam':
    cap = cv2.VideoCapture(os.path.abspath(args.source))
elif args.source == 'webcam':
    cap = cv2.VideoCapture(0)

#selecting a suitable haar cascade model.
faceCascade = cv2.CascadeClassifier('models//haarcascade_frontalface_alt2.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
cap.set(cv2.CAP_PROP_FPS, int(args.fps))


# function to capture frames from a video or a webcam.
def getdata():
    grabbed, fr = cap.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.25, 4)
    return faces, fr, gray

# function to check whether recorded face regions overlap with the new ones.
def overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if (x2 >= w1 or x1 >= w2):
        return False
    if (y2 >= h1 or y1 >= h2):
        return False

    return True

# if new face region overlaps with the rrecorded one,
# then this functions calculates the area of ovelapping.
def areaOfIntersection(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    aoi = (min(w1, w2) - max(x1, x2)) * (min(h1, h2) - max(y1, y2))
    rect1Area = (w1 - x1) * (h1 - y1)
    prcntAOI = (aoi / rect1Area) * 100
    return prcntAOI

# this function checks if the face lies in the recorded face region.  
def liesInside(rect, pt):
    p, q = pt
    x, y, w, h = rect
    if (p <= w and p >= x and q <= h and p >= y):
        return True
    else:
        return False

# This is where actual magic happens
# This function drives the entire app. 

def start_app(cnn):
    # This is the dictionary where new faces that appear in the frame are stored in the form of co-ordinates.
    # also related data is also stored.
    boxes = {}

    # This list represents the columns in the excel sheet.
    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y', 'Z',
               'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM'
               ]

    # to keep track of the columns in excel sheet.
    clm = 1
    # counts the frame no.
    frcnt = 0
    # no of recorded unique faces.
    face_cnt = 0
    
    erf = 4 # excel row where frame count starts.

    #excel sheet object.
    book = Workbook()
    sheet = book.active
    sheet['A1'] = "frameNo"

    
    while cap.isOpened():
        faces, fr, gray_fr = getdata()
        cv2.putText(fr, "Frame " + str(frcnt), (20, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # This line writes the frame no in the "frameNo" column of the excel sheet.
        sheet['A'+str(frcnt+4)] = frcnt

        # If the dictionaryy of recrded faces is empty;
        # new faces that are identified in the first frame itself are added in the dictionary.
        if len(boxes) == 0:
            for (x, y, w, h) in faces:
                crop_face = fr[y:y + h, x:x + w]
                x_o, y_o, w_o, h_o = x - 0, y - 0, x + w + 0, y + h + 0
                name = 'face'+str(face_cnt)

                # New face is being stored in the dictionary with the details like;
                # face-co-ordinates and counts of the emotions which are updated in the each frame.
                boxes[(x_o, y_o, w_o, h_o)] = {"Frustrated": 0, "Disagree": 0,
                                               "Tense": 0, "Happy": 0,
                                               "Surprise": 0, "Neutral": 0,
                                               "show_img":crop_face,"face_name":name}


                # this line writes face's co-ordinates in the sheet row-wise
                sheet[columns[clm]+ str(1)] = str((x_o, y_o, w_o, h_o))

                # this line writes face's unique name such as 'face0' in the sheet row-wise
                sheet[columns[clm] + str(2)] = name


                # this code writes face's image in the sheet row-wise
                cv2.imwrite(name+'.png',crop_face)
                img_in_sheet = Image(name+'.png')
                sheet.add_image(img_in_sheet,columns[clm] + str(3) )
                clm+=1
                face_cnt += 1

        else:
            
            for (x, y, w, h) in faces:
                isNew = True
                crop_face = fr[y:y + h, x:x + w]
                x_o, y_o, w_o, h_o = x - 0, y - 0, x + w + 0, y + h + 0
                box_length = len(boxes)


                # This loop traverses through the recorded faces' dictionary
                # and checks if newly detected face overlaps with the any of the recorded faces.
                # if the newly detected face overlaps with any of the face then isNew flag is changed to false.
                for (p, q, r, s) in boxes:
                    ifo = p_new, q_new, r_new, s_new = p - 25, q - 25, p + 25, q + 25

                    if (overlap((x_o, y_o, w_o, h_o),(p,q,r,s))):
                        if(areaOfIntersection((x_o, y_o, w_o, h_o),(p,q,r,s)) > 20):
                            isNew = False


                # If face is new it is added to the recorded faces' dictionary

                if (isNew):
                    name = 'face' + str(face_cnt)
                    boxes[(x_o, y_o, w_o, h_o)] = {"Frustrated": 0, "Disagree": 0,
                                                   "Tense": 0, "Happy": 0,
                                                   "Surprise": 0, "Neutral": 0,
                                                   "show_img":crop_face,"face_name":name}
                    sheet[columns[clm]+str(1)] = str((x_o, y_o, w_o, h_o))
                    sheet[columns[clm] + str(2)] = name
                    cv2.imwrite(name + '.png', crop_face)
                    img_in_sheet = Image(name + '.png')
                    sheet.add_image(img_in_sheet, columns[clm] + str(3))
                    clm += 1
                    face_cnt+=1



        # This code draws a box around detected face and writes its name on the frame.
        for each in boxes:
            cv2.rectangle(fr, (each[0], each[1]), (each[2], each[3]), (0, 255, 0), 1)
            cv2.putText(fr, boxes[each]['face_name'], (each[2]-40, each[3]+20 ),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)

            

        # This code writes predicted emotions for each face on the top of it.
        for (x, y, w, h) in faces:
            face = (x,y,w,h)
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            face_new = face[0], face[1], face[0] + face[2], face[1] + face[3]
            for a_face in boxes:
                if overlap(face_new,a_face):
                    for ln in range(len(boxes)):
                        
                        # predicted emotion is being written in the excel sheet.
                        if(sheet[columns[ln+1]+str(2)].value == boxes[a_face]["face_name"]):
                            sheet[columns[ln+1]+str(erf)] = pred


            cv2.putText(fr, pred, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 255, 0), 1)


        cv2.imshow('Facial Emotion Recognition', fr)
        frcnt += 1
        erf +=1

    # saves the excel sheet.
    book.save("results2.xlsx")
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    model = FacialExpressionModel("model.json", "weights.h5")
    start_app(model)
