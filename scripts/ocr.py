import os
import io
import easyocr
from PIL import Image, ImageDraw

EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "models", "EasyOCR")

print("begin to load easy ocr model,path is %s" % (MODEL_PATH))
reader = easyocr.Reader(lang_list=['ch_sim', 'en'], model_storage_directory=os.path.join(MODEL_PATH, 'model'),
                        download_enabled=True,
                        detector=True, recognizer=False,
                        user_network_directory=os.path.join(MODEL_PATH, 'user_network'), detect_network='craft')
print("finish to load easy ocr model,path is %s" % (MODEL_PATH))


def get_mask_from_file(img):
    # horizontal_list is a list of regtangular text boxes. The format is [x_min, x_max, y_min, y_max].
    # free_list is a list of free-form text boxes. The format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    height = img.height
    width = img.width
    mode = img.mode
    with io.BytesIO() as output_bytes:
        img.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
    horizontal_list, free_list = reader.detect(bytes_data)
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
    for cooridate in cooridate_list:
        mask_draw.rectangle(xy=(cooridate["x_min"], cooridate["y_min"], cooridate["x_max"], cooridate["y_max"]),
                            fill=mask_color)
    return mask
