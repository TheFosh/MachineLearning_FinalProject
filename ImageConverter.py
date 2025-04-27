import os
import matplotlib.image
import numpy as np
import imageio.v2 as imageio

train_direct = "ImageData/train"

allCardDirect = os.listdir(train_direct)
card = "ImageData/train/ace of clubs/001.jpg"
img = (imageio.imread(card, mode="F"))/255
img = np.array(img)
#np.savetxt("CSVData/image.csv", img, delimiter=",")

matplotlib.image.imsave('name.png', img)
# for card_direct in allCardDirect:
#     card_files = os.listdir(card_direct)
#     for card in card_files:
#         img = (imageio.imread(card, as_gray=True))/255
#         img = np.array(img)
#
#         np.savetxt("CSVData/" + card + ".csv", img, delimiter=",")