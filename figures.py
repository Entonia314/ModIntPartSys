import imageio
filenames = []
for n in range(0, 1240, 10):
    filenames.append(f"figure/frame{n}.png")
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('C:/Users/AltonV/PycharmProjects/ModIntPartSys/figure/nbody.gif', images)