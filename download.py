import wget
import tempfile
import os

url = 'https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/pytorch_model-000{}-of-00046.bin'



# # 下载文件，使用默认文件名,结果返回文件名
# file_name = wget.download(url)
# print(file_name) #1106F5849B0A2A2A03AAD4B14374596C76B2BDAB_w1000_h626.jpg


# 下载文件，重新命名输出文件名
# target_name = 't1.jpg'
i=18
while True:
    try:
        path = os.path.join("/data1/lileilai/gpt_20B/")
        print("idx: ", i)
        print(url.format(i))
        file_name = wget.download(url.format(i, i), out=path)
        i = i + 1
    except Exception as e:
        print(e)
        continue
# print(file_name) #t1.jpg

# # 创建临时文件夹，下载到临时文件夹里
# tmpdir = tempfile.gettempdir()
# target_name = 't2.jpg'
# file_name = wget.download(url, out=os.path.join(tmpdir, target_name))
# print(file_name)  #/tmp/t2.jpg
