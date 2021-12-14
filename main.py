import cv2 as cv
import math

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-p","--path", help="Path to image(jpg,jpeg,png) or video(mpeg,mp4,avi)",required=True)
parser.add_argument("-v","--video",action='store_true', help="Flag indicating the path is a video",default=False)
parser.add_argument("-s","--smile",action='store_true', help="Flag indicating to detect smiles rather than only faces and eyes",default=False)


args = parser.parse_args()



# loading haar cascade filters
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv.CascadeClassifier("haarcascade_smile.xml")

FACE_RECT_COLOR = (255,0,0)
EYE_RECT_COLOR = (0,255,0)
SMILE_RECT_COLOR = (0,0,255)
RECT_THICKNESS = 2


def face_detection(gray_img, img):
    faces = face_cascade.detectMultiScale(gray_img,1.3,5)
    # eyes = eye_cascade.detectMultiScale(gray_img,1.3,5)

    for x,y,w,h in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), FACE_RECT_COLOR, RECT_THICKNESS)
        face_patch_gray = gray_img[y:y+h,x:x+w]
        face_patch = img[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(face_patch_gray,1.3,5)
        for ex,ey,ew,eh in eyes:
            cv.rectangle(face_patch, (ex,ey), (ex+ew,ey+eh), EYE_RECT_COLOR, RECT_THICKNESS)

def smile_detection(gray_img, img):
    faces = face_cascade.detectMultiScale(gray_img,1.3,5)
    # eyes = eye_cascade.detectMultiScale(gray_img,1.3,5)

    for x,y,w,h in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), FACE_RECT_COLOR, RECT_THICKNESS)
        face_patch_gray = gray_img[y:y+h,x:x+w]
        face_patch = img[y:y+h,x:x+w]
        smiles = smile_cascade.detectMultiScale(face_patch_gray,1.3,5)
        for ex,ey,ew,eh in smiles:
            cv.rectangle(face_patch, (ex,ey), (ex+ew,ey+eh), SMILE_RECT_COLOR, RECT_THICKNESS)


media_path = args.path

if(args.video):
    print("Video being processed")
    cap = cv.VideoCapture(media_path)
    opened = cap.isOpened()

    if(opened):
        fps = cap.get(cv.CAP_PROP_FPS)
        print('Frames per second : ', fps, 'FPS')
        frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        print('Frame count : ', frame_count)

        delay_ms_frame = math.ceil(1/(fps/1000)) 
        print('Delay per frame (ms) : ', delay_ms_frame)

        while opened:
            ret, frame = cap.read()
            if(ret):
                gray_img = cv.cvtColor(frame,cv.IMREAD_GRAYSCALE)
                if not args.smile:
                    face_detection(gray_img,frame)
                else:
                    smile_detection(gray_img,frame)
                cv.imshow("frames", frame)
                if cv.waitKey(delay_ms_frame) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv.destroyAllWindows()
    else:
        print("Not a proper video file given in the path argument")
    
else:
    img = cv.imread(media_path,cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print("Image being processed")
    if not args.smile:
        face_detection(gray_img,img)
    else:
        smile_detection(gray_img,img)

    cv.imshow("frame",img)
    cv.waitKey()