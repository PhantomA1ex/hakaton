import cv2
import numpy
import pytesseract
import os
import PIL
from PIL import Image, ImageEnhance, ImageFilter

os.putenv("TESSDATA_PREFIX", "Path")
os.environ['TESSDATA_PREFIX'] = "C:\\Program Files\\Tesseract-OCR"

path = "img\\text-img.png"

testdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.TESSDATA_PREFIX = r"C:\\Program Files\\Tesseract-OCR"
custom_config = r'--oem 3 --psm 6'
number_plate_cascade = cv2.CascadeClassifier(r'C:\Users\seryy\PycharmProjects\pythonProject2\haarcascade_russian_plate_number(1).xml')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #frame = numpy.array(frame)

    # Преобразуем в черно-белый рисунок:
    #thresh = 200
    #fn = lambda x: 255 if x > thresh else 0
    #im = Image.fromarray(numpy.uint8(frame))
    #res = im.convert('L').point(fn, mode='1')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    number_plates = number_plate_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5, minSize=(100,100))
    for (x, y, w, h) in number_plates:
        number_plate = gray[y:y + h, x:x+w]
        number = pytesseract.image_to_string(number_plate, lang='RUS', config=custom_config)
        #number = pytesseract.image_to_string(number_plate)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, number, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Number Plate Recognition', frame)
        print(number)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cap.release()
            cap.destroyAllWindows()
