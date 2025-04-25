import numpy as np
import imageio


img = (imageio.imread(img_location, as_gray=True))/255
img = np.array(img)

np.savetxt("image.csv", a, delimiter=",")