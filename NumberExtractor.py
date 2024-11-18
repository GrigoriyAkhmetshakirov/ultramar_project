import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint
import shutil
import sys
from datetime import datetime


import cv2
from ultralytics import YOLO

from paddleocr import PaddleOCR

class NumberExtractor:
    def __init__(self, 
                 number_searching_model_path=None, 
                 horizontal_model_path=None,
                 det_model_path=None,
                 rec_model_path=None,
                 cls_model_path=None,
                 ocr_lang='en'):
        
        self.models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models_for_search'))

        if number_searching_model_path is None:
            number_searching_model_path = os.path.join(self.models_directory, 'new_number_searching_model.pt')
        if horizontal_model_path is None:
            horizontal_model_path = os.path.join(self.models_directory, 'symbol_searching_model.pt')
        if det_model_path is None:
            det_model_path = os.path.join(self.models_directory, 'det', 'en', 'en_PP-OCRv3_det_infer')
        if rec_model_path is None:
            rec_model_path = os.path.join(self.models_directory, 'rec', 'en', 'ch_ppocr_server_v2.0_rec_train')
        if cls_model_path is None:
            cls_model_path = os.path.join(self.models_directory, 'rec', 'en', 'ch_ppocr_mobile_v2.0_cls_infer')

        self.number_searching_model = YOLO(number_searching_model_path)
        self.horizontal_model = YOLO(horizontal_model_path)
        self.ocr_reader = PaddleOCR(lang=ocr_lang,
                                    show_log=False,
                                    use_angle_cls=True,
                                    det_model_dir=det_model_path,
                                    rec_model_dir=rec_model_path,
                                    cls_model_dir=cls_model_path,
        )

        self.classes = {0: 'check_digit', 1: 'number_h', 2: 'number_v', 3: 'number_v_dig', 4: 'number_v_lett'}
        self.alphabet = {**{'U': 32, 'M': 24, 'T': 31, 'L': 23}, **{str(x): x for x in range(10)}}
        self.predicted_numbers_count = 0 # Всего номеров распознано
        self.correctly_numbers_count = 0 # Количество верно распознанных номеров
        self.save_path = 'bad_images'
        os.makedirs(self.save_path, exist_ok=True)

    def parse_predictions(self, image_path, show_cropped=False, show_predictions=False, model=None, conf=0.6, verbose=True):
        try:
            image = cv2.imread(image_path)
        except:
            image = image_path
        if model is None:
            model = self.number_searching_model
        cropped_images = []
        predictions = model.predict(image, show=False, verbose=False, conf=conf)
        if show_predictions:
            predictions[0].show()
        for prediction in predictions:
            for box in prediction.boxes:
                x1, y1, x2, y2, _, cls = box.numpy().data[0]
                cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                cropped_images.append([self.classes.get(cls), cropped_image])
                if show_cropped:
                    plt.imshow(cropped_image)
                    plt.show()
        if len(cropped_images) == 0:
            cprint('No cargo number detected', 'red', 'on_black') if verbose else None
            return None
        else:
            cprint(f'Detected cargo number types is {[x[0] for x in cropped_images]}', 'green', 'on_black') if verbose else None
            return cropped_images
    
    def get_horizontal_number(self, image, h=640, w=320, show_horizontal=False, show_original=False, conf=0.75, show_predicted=False, verbose=False):
        height, width, _ = image.shape
        if height > width:
            cprint('Vertical image is detected', 'green', 'on_black') if verbose else None
            predictions = self.horizontal_model.predict(image, show=show_predicted, save_crop=False, conf=conf, verbose=False)
            boxes = predictions[0].boxes.numpy().data
            if len(boxes) not in  [11, 7, 4]:
                cprint('Not enough symbols detected', 'red', 'on_black') if verbose else None
                return None
            res = np.zeros((h, 1, 3), np.uint8)
            # line = np.zeros((h, 10, 3), np.uint8)
            for box in sorted(boxes, key=lambda x: x[1]):
                x1, y1, x2, y2, conf, cls = box
                # x1, y1, x2, y2, conf, cls = boxes[n]
                imgCrop = image[int(y1):int(y2), int(x1):int(x2)]
                res = cv2.hconcat([res, cv2.resize(imgCrop, (w, h))])
                # return res
        else:
            cprint('Horizontal image is detected', 'green', 'on_black') if verbose else None
            res = image
        if show_original:
            cprint('Original image is:', 'green', 'on_black')
            plt.axis('off')
            plt.imshow(image)
            plt.show()
        if show_horizontal:
            cprint('Horizontal number is:', 'green', 'on_black')
            plt.axis('off')
            plt.imshow(res)
            plt.show()
        return res
    
    def get_text(self, file):
        try:
            return self.ocr_reader.ocr(file)
        except:
            print('No text on the image')
            return None

    def get_cargo_number(self, 
                         file, 
                         show_horizontal=False, 
                         show_original=False, 
                         show_prepared=False, 
                         print_pred_text=False, 
                         show_predictions=False,
                         show_cropped=False, 
                         thresh_horizontal_number=0,
                         thresh_vertical_number=1,
                         thresh_check_digit=1,
                         blure_horizontal_number=1,
                         blure_vertical_number=7,
                         blure_check_digit=7,
                         size_number=(640, 160),
                         size_check_digit=(200, 200),
                         verbose=True,
                         save_bad_image=True,
                         get_metrics=True):
        cropped_images = self.parse_predictions(file, show_cropped=show_cropped, show_predictions=show_predictions, verbose=verbose)
        self.get_accuracy(verbose=verbose) if get_metrics else None
        if cropped_images is None:
            cprint(f'No images to work', 'red', 'on_black') if verbose else None
            return None
        try:
            number_types = [x[0] for x in cropped_images]
            for image in cropped_images:
                if image[0] == 'number_v':
                    number = self.get_horizontal_number(image[1], show_horizontal=show_horizontal, show_original=show_original, verbose=verbose)
                    prepared_image = self.prepare_image(number, thresh=thresh_vertical_number, blure=blure_vertical_number, show=show_prepared, resize=True, size=size_number)
                    text = self.get_text(prepared_image)[0][0][1][0]
                    if print_pred_text:
                        cprint(f'Text after OCR is {text}', 'blue')

                if image[0] == 'number_h':
                    number = self.get_horizontal_number(image[1], show_horizontal=show_horizontal, show_original=show_original, verbose=verbose)
                    prepared_image = self.prepare_image(number, thresh=thresh_horizontal_number, blure=blure_horizontal_number, show=show_prepared, resize=True, size=size_number)
                    texts = self.get_text(prepared_image)
                    if print_pred_text:
                        cprint(f'Text after OCR is {texts}', 'blue')
                    text_list = []
                    for el in texts[0]:
                        if len(el[1][0]) in [4, 6]:
                            text_list.append(el[1][0])
                    text = ''.join(sorted(text_list, reverse=True))
                    if print_pred_text:
                        cprint(f'text after preparing is {text}', 'blue')

                if image[0] == 'check_digit' and 'number_h' in number_types:
                    prepared_image = self.prepare_image(image[1], thresh=thresh_check_digit, blure=blure_check_digit, show=show_prepared, resize=True, size=size_check_digit)
                    try:
                        check_digit_text = self.get_text(prepared_image)[0][0][1][0]
                        # print(check_digit_text)
                        for el in check_digit_text:
                            if el.isdigit():
                                check_digit = el
                                break
                        cprint(f'Detected check digit is: {check_digit}', 'blue') if verbose else None
                    except:
                        check_digit = None
                        cprint('No check digit detected', 'red', 'on_black') if verbose else None
                    

                if image[0] == 'number_v_dig':
                    number_v_dig = cv2.resize(image[1], (90, 600))
                
                if image[0] == 'number_v_lett':
                    number_v_lett = cv2.resize(image[1], (90, 300))
            
            if 'number_v_lett' in number_types and 'number_v_dig' in number_types:
                collected_number = cv2.vconcat([number_v_lett, number_v_dig])
                number = self.get_horizontal_number(collected_number, show_horizontal=show_horizontal, show_original=show_original, verbose=verbose)
                prepared_image = self.prepare_image(number, thresh=1, blure=7, show=show_prepared, resize=True, size=(640, 160))
                text = self.get_text(prepared_image)[0][0][1][0]

            if text[2] == 'T' and len(text) == 11:
                text = 'UMTU' + text[4:10] + text[-1]

            if text[2] == 'L' and len(text) == 11:
                text = 'UMLU' + text[4:10] + text[-1]
                
            if text[2] == 'L' and len(text) != 11 and check_digit:
                text = 'UMLU' + text[4:10] + check_digit

        except Exception as e:
            cprint(e, 'red', 'on_black')
            return None
        cprint(f'Detected text is {text}', 'blue') if verbose else None
        self.predicted_numbers_count += 1
        if len(text) == 11 and text[0:4].isalpha() and text[4::].isdigit() and self.check_number(text, verbose=verbose):
            self.correctly_numbers_count += 1
            return text
        else:
            if save_bad_image:
                if isinstance(file, str):
                    shutil.copy(file, self.save_path)
                    save_name = os.path.basename(file)
                else:
                    save_name = f'{text} {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
                    image_save_path = os.path.join(self.save_path, save_name)
                    cv2.imwrite(image_save_path, file)
                cprint(f'Bad image {save_name} saved in {self.save_path}', 'yellow') if verbose else None
            cprint(f'Prepared text {text} is wrong', 'red', 'on_black') if verbose else None
            return None
    
    def get_auto_number(self, file, print_text=False):
        text = self.get_text(file)
        if print_text:
            print(f'Detected text is: {text}')
        try:
            text = [text[0][x][1][0] for x in range(len(text[0]))]
            for el in text:
                if el[0:2].isalpha() and el[-3:].isdigit():
                    return el
            print('No auto number detected')
        except:
            print('No auto number detected')
        return None 
        
    def get_hopper_number(self, file, print_text=False):
        text = self.get_text(file)
        if print_text:
            print(f'Detected text is: {text}')
        try:
            text = [text[0][x][1][0] for x in range(len(text[0]))]
            for el in text:
                el = el.replace(' ', '')
                if el[0:8].isdigit() and len(el) == 8:
                    return el[0:8]
            print('No hopper number detected')
        except:
            print('No hopper number detected')
        return None 
    
    def prepare_image(self, image_path, size=(640, 160), resize=False, rgb=False, gray=False, show=False, thresh=1, blure=15):
        try:
            image = cv2.imread(image)
        except:
            image = image_path
        if resize:
            image = cv2.resize(image, size)
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if gray or thresh:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if thresh:
            image = cv2.medianBlur(image, blure)
            _, image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        if show:
            cprint('Prepared image is:', 'green', 'on_black')
            plt.axis('off')
            plt.imshow(image)
            plt.show()
        return image
    
    def check_number(self, number, verbose=False):
        res = 0
        for i, sym in enumerate(number[:-1]):
            res = res + 2**i * self.alphabet.get(sym, 0)
        res = res % 11 % 10
        cprint(f'Calculated check digit is: {res}', 'blue') if verbose else None
        return str(res) == number[-1]
    
    def get_accuracy(self, predicted_numbers_count=None, correctly_numbers_count=None, verbose=False):
        if predicted_numbers_count is None:
            predicted_numbers_count=self.predicted_numbers_count
        if correctly_numbers_count is None:
            correctly_numbers_count = self.correctly_numbers_count
        if predicted_numbers_count != 0:
            res = correctly_numbers_count / predicted_numbers_count
            cprint(f'Predicted {predicted_numbers_count} numbers, correctly predicted {correctly_numbers_count} numbers, accuracy is {res:.3}', 'magenta') if verbose else None
        else:
            res = 0
        return res
        
