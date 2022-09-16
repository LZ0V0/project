import numpy as np
import shutil
import re
import cv2

bad_list = ['6833728', '6983175', '6983176', '6990173', '7180665', '7259429', '7306127', '7516517', '7520444',
            '7566046', '7566046', '7633507', '7802216', '7878194', '7920058', '7964176', '8008994', '8122618',
            '8137703', '8303559', '8303561', '8326768', '8326770', '8353030', '8353031', '8387005', '8405721',
            '8405722', '8460296', '8494824', '8564737', '8594286', '8594530', '8594533', '8680441', '8711349',
            '8981017', '9041322', '9041323', '9058500', '9083441', '9110395', '9110396', '9122773', '9128236',
            '9155294', '9155324', '9169310', '9169311', '9190769', '9190772', '9190774', '9209594', '9209612',
            '9209613', '9338973', '9339204', '9376234', '9404782', '9416695', '9576158', '9675374', '9721149',
            '7005012', '7566077']   # images not in regular form

descs = np.load('coin_location.npy', allow_pickle=True).item()
describes_prepared = {}
num_count = {'Phoenicia': 0, 'Mesopotamia': 0, 'Syria': 0, 'Cyprus': 0, 'Cappadocia': 0, 'Judaea': 0, 'Egypt': 0}
for num, desc in descs.items():
    if re.findall('|'.join(bad_list), num) == []:   # not in the bad list
        # shutil.copy('./imgs_with_loc/{}.jpg'.format(num), './imgs_prepared/{}.jpg'.format(num))  # resave images
        describes_prepared['{}'.format(num)] = desc
        # 调整大小   灰度图
        img = cv2.imread('./imgs_with_loc/{}.jpg'.format(num), cv2.IMREAD_GRAYSCALE)  # gray scale imgs
        cv2.imwrite('./imgs_prepared/{}.jpg'.format(num), cv2.resize(img, (240, 120)))  # size adjust
        # 计数图像location分布功能
        num_count['{}'.format(desc)] += 1  # count distributions of each location

np.save('describes_prepared.npy', describes_prepared)
print(num_count)  # print location distributions

# {'Phoenicia': 269, 'Mesopotamia': 55, 'Syria': 419, 'Cyprus': 19, 'Cappadocia': 1 ??, 'Judaea': 84, 'Egypt': 9}

# 大图片链接 https://www.acsearch.info/image.html?id=9769704