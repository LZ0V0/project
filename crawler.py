import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
import numpy as np

# from selenium.webdriver.edge.options import Options
#
# edge_options = Options()
# edge_options.add_argument("--headless")       # define headless

diver_path = 'E:\Selenium driver\msedgedriver.exe'  # the path of edge selenium drive. Need  download manually
driver = webdriver.Edge(executable_path=diver_path)   # using Edge
# driver = webdriver.Edge(executable_path=diver_path, options=edge_options)  # set opinion with headless

url = 'https://www.acsearch.info/search.html?term=Caracalla+Tetradrachm+&category=1&lot=&thesaurus=1&images=1&en=1&de=1&fr=1&it=1&es=1&ot=1&currency=gbp&order=1'
# html = requests.get(url).text
driver.get(url)
# driver.find_element(By.CLASS_NAME, 'lot-date').send_keys(Keys.END)
for i in range(15):    # load more data in the web page
    ActionChains(driver).send_keys(Keys.END).perform()   # keyboard input end key
    time.sleep(5)      # wait for request form web

html = driver.page_source   # get html code
soup = BeautifulSoup(html, 'lxml')

imgs = soup.find_all('div', {'class': 'lot-img'})[1:]     # the first result is empty, start with second one

descs = soup.find_all('div', {'class': 'lot-desc'})[1:]


# img_url_title = 'https://www.acsearch.info/'
img_url_title = 'https://www.acsearch.info/image.html?id='
# 大图片链接 https://www.acsearch.info/image.html?id=9769704

# print(len(imgs))
# print(imgs)
# print(len(descs))
# print(descs)
# print(type(html))

desc_number = 0
desc_dic = {}

img_num_count = 0
desc_num_count = 0

# print(len(imgs))

for i in imgs:
    img = i.find_all('img')[0]
    # ur = img.get_text()

    # zz = descs[desc_number].find_all('span')[0].get_text()
    # print(descs[desc_number].content)
    # print(type(descs[desc_number].content))
    # print(img['src'][-3:])
    # print(str(img['src']))
    # print(type(str(img['src'])))

    # get url of coin image
    if img['src'][-3:] == 'jpg':    # find url ends with jpg
        # img_url = img_url_title + img['src']
        img_url = img_url_title + img['src'][-13:-6]
    else:
        # img_url = img_url_title + img['data-src']
        img_url = img_url_title + img['data-src'][-13:-6]

    coin_number = img_url[-13:-6]  # select the img number as name

    # download image
    print('下载 {}'.format(img_url))  # print downloading
    try:
        r = requests.get(img_url, timeout=5)  # set timeout 5 seconds
    except:
        print('下载 {} 超时, 重试'.format(img_url))  # timeout retry
        try:
            r = requests.get(img_url, timeout=5)
        except:
            print('下载 {} 超时, 放弃'.format(img_url))  # timeout again, give up
            continue
    # with open('./imgs/{}.jpg'.format(coin_number), 'wb') as f:
    with open('./imgs_larger/{}.jpg'.format(coin_number), 'wb') as f:
        f.write(r.content)   # save img
    print('下载完成')   # print saved
    img_num_count = img_num_count + 1   # count the total number of img downloaded

    # get describe of this coin and add in dictionary
    print('收集 {} 描述'.format(coin_number))
    coin_desc = descs[desc_number].find_all('span')[0].get_text()   # collect description of this img
    desc_dic['{}'.format(coin_number)] = coin_desc  # save in dictionary

    desc_number = desc_number + 1   # count the total number of description downloaded
    print('收集完成\n')
    desc_num_count = desc_num_count + 1

np.save('coin_describe.npy', desc_dic)
print('收集 {} 张图片\n收集 {} 条描述'.format(img_num_count, desc_num_count))   # print total number

