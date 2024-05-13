import os
import threading

import torch
from litelama import LiteLama
from litelama.model import download_file

EXTENSION_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(EXTENSION_PATH, "models", "lama")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def clean_object_init_img_with_mask(init_img_with_mask):
    mask_img = init_img_with_mask['mask']
    result = clean_object(init_img_with_mask['image'], init_img_with_mask['mask'])
    result.append(mask_img)
    return result


class LiteLama2(LiteLama):
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self, checkpoint_path=None, config_path=None):
        if not hasattr(self, "_model"):
            with LiteLama2._instance_lock:
                if not hasattr(self, "_model"):
                    self._checkpoint_path = checkpoint_path
                    self._config_path = config_path
                    self._model = None

                    if self._checkpoint_path is None:
                        checkpoint_path = os.path.join(MODEL_PATH, "big-lama.safetensors")
                        if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
                            pass
                        else:
                            download_file(
                                "https://aod.cos.tx.xmcdn.com/storages/anyisalin/big-lama/big-lama.safetensors",
                                checkpoint_path)

                        self._checkpoint_path = checkpoint_path

                    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
                    print("begin to load lama model,path is %s" % (self._checkpoint_path))
                    self.load(location=torch_device)
                    print("finish to load lama model,path is %s" % (self._checkpoint_path))


cleaner = LiteLama2()


def clean_object(image, mask):
    init_image = image
    mask_image = mask

    init_image = init_image.convert("RGB")
    mask_image = mask_image.convert("RGB")

    result = None
    try:
        result = cleaner.predict(init_image, mask_image)
    except:
        pass

    return [result]
