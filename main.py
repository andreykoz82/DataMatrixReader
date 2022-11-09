from dbr import *
import cv2
from pyzxing import BarCodeReader
from time import time
from scripts.helper_functions import preprocess, text_decompose

cam = cv2.VideoCapture(0)

cv2.namedWindow("Агрегация")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Ошибка чтения с камеры")
        break
    cv2.imshow("Агрегация", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Нажата клавиша ESC, завершение работы...")
        break
    elif k % 256 == 32:
        # SPACE pressed

        start = time()

        image = preprocess(frame, contrast_value=3)
        image.save('temp/box_cropped.png')

        license_key = "t0068fQAAALAuic8BW9jDzdXFloXo4Ef6M5crw8YyTYaR0V4i9+BWkeMC6kpivmFA34fwSLonUdAKN6N5JrJOGjS6WEucGqI="
        BarcodeReader.init_license(license_key)
        reader = BarcodeReader()

        try:
            text_results = reader.decode_file('temp/box_cropped.png')

            if text_results != None:
                for text_result in text_results:
                    print("Localization Points : ", text_result.localization_result.localization_points)
                    text_decompose(text_result.barcode_text)
        except BarcodeReaderError as bre:
            print(bre)

        xy, w, h, s, = text_result.localization_result.localization_points

        area = (xy[0] - 15, xy[1] - 15, h[0] + 15, h[1] + 15)
        cropped_img = image.crop(area).convert("L").resize((190, 190))
        cropped_img.save('box/datamarix.png')

        reader = BarCodeReader()

        try:
            results = reader.decode('box/*.png')
            s = results[0]['raw']
            s = s.decode("utf-8")
            text_decompose(s)
        except KeyError:
            print('Decode FAILED, try again!')

        print("Время работы: ", round(time() - start, 1))

cam.release()

cv2.destroyAllWindows()

