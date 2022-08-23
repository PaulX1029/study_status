from qiniu import Auth, put_file
from qiniu import BucketManager
from qiniu import CdnManager

# 配置七牛云信息
access_key = "HoXmloGu85nrmHm089U05rAg1MTQcBl-5Yn1tpSx"
secret_key = "l1q_-M24NCZHDwwZ16jKTtXDlCf6yvpGwhnmVHJh"
bucket_name = "aidlux-xzw"
bucket_url = "rgpgm392c.bkt.clouddn.com"
q = Auth(access_key, secret_key)
bucket = BucketManager(q)

cdn_manager = CdnManager(q)

# 将本地图片上传到七牛云中
def upload_img(bucket_name, file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name, 3600)
    put_file(token, file_name, file_path)

# 获得七牛云服务器上file_name的图片外链
def get_img_url(bucket_url, file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url

# # 需要上传到七牛云的图片路径
# image_up_path = "zidane.jpg"
# # 上传到七牛云后，保存成的图片名称
# image_qiniu_name = "zidane.jpg"
# # 将图片上传到七牛云，并保存成image_qiniu_name的名称
# upload_img(bucket_name, image_qiniu_name, image_up_path)
# # 取出image_qiniu_name一样名称图片的url
# url_receive = get_img_url(bucket_url, image_qiniu_name)
# print(url_receive)
# # 需要刷新的文件链接，由于不同时间段上传的图片有缓存，因此需要CDN清除缓存
# urls = [url_receive]
# # URL刷新缓存链接，一天有500次的刷新缓存机会
# refresh_url_result = cdn_manager.refresh_urls(urls)

def upload_qiniuyun(image_name, image_path):
    # 将图片上传到七牛云，并保存成image_qiniu_name的名称
    upload_img(bucket_name, image_name, image_path)
    # 取出image_qiniu_name一样名称图片的url
    url_receive = get_img_url(bucket_url, image_name)
    # 需要刷新的文件链接，由于不同时间段上传的图片有缓存，因此需要CDN清除缓存
    urls = [url_receive]
    # URL刷新缓存链接，一天有500次的刷新缓存机会
    refresh_url_result = cdn_manager.refresh_urls(urls)
    return url_receive