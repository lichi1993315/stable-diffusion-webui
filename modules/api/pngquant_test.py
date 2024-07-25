import pngquant
from PIL import Image
from io import BytesIO
import os
from rembg import new_session
import onnxruntime as rt
from rembg_mine import my_remove

providers = [
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider'
]


def return_provider():
    return providers


rt.get_available_providers = return_provider
print(rt.get_available_providers())

#session1 = new_session("isnet-anime")
#session2 = new_session("isnet-general-use")
#session1 = new_session("isnet-anime", rt.get_available_providers())
#print("session 1 init over", flush=True)
#session2 = new_session("isnet-general-use", rt.get_available_providers())
#print("session 2 init over", flush=True)

sessions = []


pngquant.config('/snap/bin/pngquant', min_quality=90, max_quality=90, ndeep=1, ndigits=12)
img = Image.open('20240105-231535.jpg')
file_size = os.path.getsize("20240105-231535.jpg")
print(f"The disk size of the image is: {file_size} bytes")
if len(sessions) == 0:
    session1 = new_session("isnet-anime", rt.get_available_providers())
    print("session 1 init over", flush=True)
    session2 = new_session("isnet-general-use", rt.get_available_providers())
    print("session 2 init over", flush=True)
    sessions.append(session1)
    sessions.append(session2)
# rmbg_image = my_remove(img,alpha_matting=True,post_process_mask=True, session=sessions)
filename = "t3.png"
# print("type",type(rmbg_image))
imgByteArr = BytesIO()
img.save(imgByteArr, format="PNG")
rmbg_image = imgByteArr.getvalue()
compressed_ratio, compressed_image = pngquant.quant_data(rmbg_image, dst=filename)
img = Image.open(BytesIO(compressed_image))
temp_path = 'temp_image.png'
img.save(temp_path)

# Get the size of the temporary file
file_size = os.path.getsize(temp_path)
print(f"The disk size of the image is: {file_size} bytes")


rmbg_image = my_remove(img,alpha_matting=True,post_process_mask=True, session=sessions)
temp_path = 'temp_image_final.png'
rmbg_image.save(temp_path)

# Get the size of the temporary file
file_size = os.path.getsize(temp_path)
print(f"The disk size of the image is: {file_size} bytes")

imgByteArr = BytesIO()
rmbg_image.save(imgByteArr, format="PNG")
rmbg_image = imgByteArr.getvalue()
compressed_ratio, compressed_image = pngquant.quant_data(rmbg_image, dst=filename)
img = Image.open(BytesIO(compressed_image))
temp_path = 'temp_image_1.png'
img.save(temp_path)

# Get the size of the temporary file
file_size = os.path.getsize(temp_path)
print(f"The disk size of the image is: {file_size} bytes")
# print(compressed_ratio, compressed_image)
# pngquant.quant_image('output.png', dst="t2.png")
# import subprocess

# def compress_png_with_pngquant(input_file, output_file, quality_range='60-80'):
#    command = ['pngquant', input_file, '--quality', quality_range, '-o', output_file]
#    subprocess.run(command, check=True)

#input_file = '~/stable-diffusion-webui/outputs/txt2img-images/2024-01-30/00000-3519811158.png'
#output_file = 'output.png'
#compress_png_with_pngquant(input_file, output_file)
