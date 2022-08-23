import time
import requests

# # 填写对应喵码
# # id = 'tW5iDu1'
# id = 'tSe50aT'
# # 填写喵提醒中，发送的消息，这里放上面前面提到的图片外链
# text = "告警图片：" + "  http://rgpgm392c.bkt.clouddn.com/zidane.jpg"
# # 时间戳
# ts = str(time.time())
# # 返回内容格式
# type = 'json'
# request_url = "http://miaotixing.com/trigger?"

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrom/87.0.4280.67'
# }

# result = requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type" + type, headers=headers)
# print(result)


def miaotixing_post(image_url):
    id = 'tSe50aT'
    text_iamge_url = "告警图片：" + image_url
    print(text_iamge_url)
    ts = str(time.time())
    type = 'json'
    request_url = "http://miaotixing.com/trigger?"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrom/87.0.4280.67'
    }
    result = requests.post(request_url + "id=" + id + "&text=" + text_iamge_url + "&ts=" + ts + "&type" + type, headers=headers)
    print(result)
