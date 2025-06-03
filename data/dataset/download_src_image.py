import os
import requests
from PIL import Image
from io import BytesIO

# 请求验证码 URL
url = "https://verify.zijieapi.com/captcha/get?aid=1402&lang=zh&bd_version=1.0.0.578&subtype=slide&detail=qzU50YPDrMWvT4YaLabsrq8mJfWpGgmyxRecpCAoDNBcq6Ovgiqb0mQXVUt5pbbSZyM11lh6Vowd4meAKUjPcMgkrwkm3W82596l6SNMIOKDFpxXHWNpze8xJ-8kQffjj17J2g95eA6qiU1*SP*RgtqrovKzeIaksXpuW39LIsqMHo*XY9-iMCDi3Z42mkZ*NOTOEFjI5A8yJoAePxFBzc7Yh1iNrIgD0N80Jra5-zbykifYON0C91Cs6m2kpP-FaeSHFC*1jaFw5zgqWOMyOmW8TeOX47Iq1h19cOkKb0*HhGeE0GE2buoXwSvDmeWGptNU436eYJ5NnndJoR0mPuFmy-9V6myFjMjeZCwRg3KRuC-Y4kLYK8zZJ5Rms-QwH9VsA07amBFoLNvAHhbu2jex7JeYdIiGpWkF2I7xm9AXZlOR6DNwFfrScMEIazojUhcNUqnY7bLV&server_sdk_env={%22idc%22:%22hl%22,%22region%22:%22CN%22,%22server_type%22:%22passport%22}&mode=slide&fp=verify_mb7oju6a_ef2f80ef_c851_beac_33bb_04bf6412b7b4&h5_check_version=4.0.5&os_name=windows&platform=pc&os_type=2&h5_sdk_version=3.5.71&webdriver=false&tmp=1748423127542"


def download_image(url, save_path):
    img_response = requests.get(url)
    if img_response.status_code == 200:
        img = Image.open(BytesIO(img_response.content))
        img.save(save_path)
        print(f"Image saved at: {save_path}")
    else:
        print(f"Failed to download image from {url}")


def request_img_address():
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()

        # 解析返回的 JSON 数据
        question = result['data']['question']
        url1 = question['url1']
        url2 = question['url2']

        # 保存图片
        download_image(url1, f'src/bg/{url1.split("/")[-1]}')
        download_image(url2, f'src/slide/{url2.split("/")[-1]}')

    else:
        print(f"Failed to retrieve captcha: {response.status_code}")



if __name__ == '__main__':
    for i in range(100):
        request_img_address()


