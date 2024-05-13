import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("litelama"):
    launch.run_pip("install litelama==0.1.7")
if not launch.is_installed("easyocr"):
    launch.run_pip("install easyocr==1.7.1")
