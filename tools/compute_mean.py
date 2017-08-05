import sys
import numpy as np
from PIL import Image
import progressbar
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as fd:
        bar = progressbar.ProgressBar()
        for line in bar(fd):
            file_name = line.split()[0]
            image = np.array(Image.open(file_name).resize((1241, 376),
                                                          Image.ANTIALIAS))
            try:
                total += image
                k += 1
            except NameError:
                total = image.astype(np.int32)
                k = 1
    mean_image = total/k
    np.savez('mean_image.npz', mean_image)
    plt.imshow(mean_image)
    plt.show()
