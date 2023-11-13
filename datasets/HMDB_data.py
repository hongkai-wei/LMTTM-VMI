import os
import requests

def download_from_baiyun(url, save_path):
    response = requests.get(url)
    response.raise_for_status()

    with open(save_path, 'wb') as file:
        file.write(response.content)

if __name__ == '__main__':
    file_url = 'https://www.123pan.com/s/wTDfjv-KW8P3.html'  # 替换为您的百度云文件链接
    save_directory = './datasets_data/HMDB51/dataset0'  # 替换为您希望保存文件的目录

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = os.path.basename(file_url)
    save_path = os.path.join(save_directory, file_name)

    if not os.path.exists(save_path):
        download_from_baiyun(file_url, save_path)
        print(f"文件已保存到：{save_path}")
    else:
        print(f"文件已存在：{save_path}")