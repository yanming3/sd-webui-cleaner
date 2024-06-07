import os
import io
from custom.easyocr import Reader
import torch
from PIL import Image, ImageDraw

EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "models", "EasyOCR")


class TextDetector:
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        is_use_gpu = torch.cuda.is_available()

        print(f"begin to load easy ocr model,path is  {MODEL_PATH},use gpu={is_use_gpu}")
        self._model = Reader(lang_list=['ch_sim', 'en'],
                                     gpu=is_use_gpu,
                                     model_storage_directory=os.path.join(MODEL_PATH, 'model'),
                                     download_enabled=True,
                                     detector=True, recognizer=False,
                                     user_network_directory=os.path.join(MODEL_PATH, 'user_network'),
                                     detect_network='craft')
        print(f"finish to load easy ocr model,path is  {MODEL_PATH},use gpu={is_use_gpu}")

    def detect(self, bytes_data):
        return self._model.detect(bytes_data)


text_detector = TextDetector()


def get_mask_from_file(img):
    # horizontal_list is a list of regtangular text boxes. The format is [x_min, x_max, y_min, y_max].
    # free_list is a list of free-form text boxes. The format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    height = img.height
    width = img.width
    mode = img.mode

    with io.BytesIO() as output_bytes:
        img.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
    horizontal_list, free_list = text_detector.detect(bytes_data)
    list = []
    for item in horizontal_list:
        for c in item:
            x_min = c[0]
            x_max = c[1]
            y_min = c[2]
            y_max = c[3]
            list.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max})
    return _make_mask(list, height, width, mode)


def _make_mask(cooridate_list, height: int, width: int, mode: str) -> Image:
    num_channels = len(mode)
    background_color = tuple([0] * num_channels)
    mask_color = tuple([255] * num_channels)

    mask = Image.new(mode, (width, height), background_color)
    mask_draw = ImageDraw.Draw(mask)
    for coordinate in cooridate_list:
        mask_draw.rectangle(xy=(coordinate["x_min"], coordinate["y_min"], coordinate["x_max"], coordinate["y_max"]),
                            fill=mask_color)
    return mask
