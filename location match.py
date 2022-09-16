import re
import numpy as np
import shutil


def location_match(original_data):  # match by keywords
    Phoenicia = ['Phoenicia', 'Phönikien', 'Phénicie', 'phénicien', 'Berytus', 'Beletus']
    Mesopotamia = ['Mesopotamia', 'Mesopotamien', 'Mésopotamie']
    Syria = ['Syria', 'Cyrrhestica', 'Seleucis', 'Pieria', 'Damascus', 'Antioch',
             'Syrien', 'Cyrrhestica', 'Seleukis', 'Pierie', 'Damaskus', 'Antiochia',
             'Syrie', 'silestika', 'Seleucia' 'pieria', 'Damas', 'antioque']
    Cyprus = ['Cyprus', 'ZYPERN', 'Chypre']
    Cappadocia = ['Cappadocia', 'Kappadokien', 'Cappadoce']
    Judaea = ['Judaea', 'Judäa', 'Juifs']
    Egypt = ['Egypt', 'Heliopolis', 'Ägypten', 'Heliopolis']
    location_list = [Phoenicia, Mesopotamia, Syria, Cyprus, Cappadocia, Judaea, Egypt]

    for loc in location_list:
        # print('loc is {}'.format(loc))
        pattern = '|'.join(loc)
        location = re.findall(pattern, original_data, flags=re.I)   # ignore case
        # print(location)
        if location != []:
            location = loc[0]  # get the str location name
            return location

    return None   # no location massage find


coin_descs = np.load('coin_describe.npy', allow_pickle=True).item()
coin_location = {}

for num, desc in coin_descs.items():
    location = location_match(desc)
    if location != None:
        coin_location['{}'.format(num)] = location
        shutil.copy('./imgs/{}.jpg'.format(num), './imgs_with_loc/{}.jpg'.format(num))  # resave images with location

print(len(coin_location))
# print(coin_location)
np.save('coin_location.npy', coin_location)

# test = coin_descs['9736093']
# print(location_match(test))
