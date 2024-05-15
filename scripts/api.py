from fastapi import FastAPI, Body

from modules.api.models import *
from modules.api import api
import gradio as gr

from custom import lama
from custom import ocr


def cleanup_api(_: gr.Blocks, app: FastAPI):
    @app.post("/sdapi/v1/cleanup")
    def clean_up(
            input_image: str = Body("", title='cleanup input image', embed=True)
    ):

        _image = api.decode_base64_to_image(input_image)
        _mask = ocr.get_mask_from_file(_image)

        _output = lama.clean_object(_image, _mask)

        if len(_output) > 0:
            return {"code": 0, "message": "ok", "image": api.encode_pil_to_base64(_output[0]).decode("utf-8")}
        else:
            return {"code": -1, "message": "Image generation failed"}

    @app.post("/sdapi/v1/cleanup_with_mask")
    def cleanup_with_mask(
            input_image: str = Body("", title='cleanup input image'),
            mask: str = Body("", title='clean up mask')
    ):

        _image = api.decode_base64_to_image(input_image)
        _mask = api.decode_base64_to_image(mask)

        _output = lama.clean_object(_image, _mask)

        if len(_output) > 0:
            return {"code": 0, "message": "ok", "image": api.encode_pil_to_base64(_output[0]).decode("utf-8")}
        else:
            return {"code": -1, "message": "Image generation failed"}


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(cleanup_api)
except:
    pass
