import matplotlib.pyplot as plt
import cv2

def no_axis_show(img, title='', cmap=None):
  # imshow, 縮放模式為nearest。
  fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
  # 不要顯示axis。
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.title(title)

if __name__ == '__main__':

    titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
    plt.figure(figsize=(18, 18))
    for i in range(10):
      plt.subplot(1, 10, i+1)
      fig = no_axis_show(plt.imread(f'train_data/{i}/{500*i}.bmp'), title=titles[i])
    plt.show()

    plt.figure()
    for i in range(10):
      plt.subplot(1, 10, i+1)
      fig = no_axis_show(plt.imread(f'test_data/{i}.bmp'))
    plt.show()

    plt.figure(figsize=(18, 18))
    original_img = plt.imread(f'train_data/0/0.bmp')
    plt.subplot(1, 5, 1)
    no_axis_show(original_img, title='original')

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    plt.subplot(1, 5, 2)
    no_axis_show(gray_img, title='gray scale', cmap='gray')

    canny_50100 = cv2.Canny(gray_img, 50, 100)
    plt.subplot(1, 5, 3)
    no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

    canny_150200 = cv2.Canny(gray_img, 150, 200)
    plt.subplot(1, 5, 4)
    no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

    canny_250300 = cv2.Canny(gray_img, 250, 300)
    plt.subplot(1, 5, 5)
    no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')
    plt.show()