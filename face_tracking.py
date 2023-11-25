import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import weightedstats as ws


class FaceTracker:
    def __init__(self, input_path, N, Nb, landa, output_path):
        self.N = N
        self.Nb = Nb
        self.landa = landa
        self.sequence_path = input_path
        self.save = output_path

    def read_image(self, tt):
        sequence = self.sequence_path
        filenames = os.listdir(sequence)
        filenames.sort(key=lambda x: int(x.split('.')[0][5:]))
        T = len(filenames)
        im = Image.open((str(sequence) + str(filenames[tt])))
        plt.imshow(im)
        return im, filenames, T, sequence

    def select_zone(self):
        print('Click 4 points on the image to define the zone to track.')
        zone = np.zeros([2, 4])
        counter = 0
        while counter != 4:
            res = plt.ginput(1)
            if not res:  # Check if ginput returned a result
                print("No input detected.")
                continue
            a = res[0]
            zone[0, counter] = a[0]
            zone[1, counter] = a[1]
            plt.plot(a[0], a[1], marker='X', color='red')
            counter += 1
            print(f"Point registered. Counter: {counter}")

        new_zone = np.zeros([2, 4])
        new_zone[0, :] = np.sort(zone[0, :])
        new_zone[1, :] = np.sort(zone[1, :])

        zoneAT = np.zeros([4])
        zoneAT[0] = new_zone[0, 0]
        zoneAT[1] = new_zone[1, 0]
        zoneAT[2] = new_zone[0, 3] - new_zone[0, 0]
        zoneAT[3] = new_zone[1, 3] - new_zone[1, 0]

        xy = (zoneAT[0], zoneAT[1])
        rect = ptch.Rectangle(xy, zoneAT[2], zoneAT[3], linewidth=3, edgecolor='red', facecolor='None')
        current_axis = plt.gca()
        current_axis.add_patch(rect)
        plt.show(block=False)
        return zoneAT

    def recreate_image(self, codebook, labels, w, h):
        d = codebook.shape[1]
        image = np.zeros((w, h))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = labels[label_idx]
                label_idx += 1
        return image

    def rgb2ind(self, im, nb):
        image = np.array(im, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(image.shape)
        image_array = np.reshape(image, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0)[:1000]

        if type(nb) == int:
            kmeans = KMeans(n_clusters=nb, random_state=0).fit(image_array_sample)
        else:
            kmeans = nb

        labels = kmeans.predict(image_array)
        image = self.recreate_image(kmeans.cluster_centers_, labels, w, h)
        return Image.fromarray(image.astype('uint8')), kmeans

    def calculate_histogram(self, im, zoneAT, Nb):
        box = (zoneAT[0], zoneAT[1], zoneAT[0] + zoneAT[2], zoneAT[1] + zoneAT[3])
        little_im = im.crop(box)
        new_im, kmeans = self.rgb2ind(little_im, Nb)
        histogram = np.asarray(new_im.histogram())
        histogram = histogram / np.sum(histogram)
        return new_im, kmeans, histogram

    def distance(self, map):
        s = 0
        for i in range(self.Nb):
            s += np.sqrt(map[i] * self.refmap[i])
        return 1 - s

    def weight(self, particles, k):
        l = []
        for i in range(self.N):
            l.append(np.exp(-self.landa * self.distance(self.calculate_histogram(self.read_image(k)[0],
                                                                                   [particles[i][0], particles[i][1],
                                                                                    self.selecz[2],
                                                                                    self.selecz[3]], self.Nb)[2])))
        return l

    def track_face(self):
        sar = self.read_image(0)
        self.selecz = self.select_zone()
        _, _, self.refmap = self.calculate_histogram(self.read_image(0)[0], self.selecz, self.Nb)
        inip = np.random.normal([self.selecz[0], self.selecz[1]], np.sqrt(300), (self.N, 2))

        w = self.weight(inip, 0)
        w = np.array(w)
        w = w / w.sum()
        w = w.flatten()
        X_l = [inip]
        unique_colors, counts = np.unique(sar[0], return_counts=True, axis=2)
        plt.clf()
        plt.ion()
        X = inip
        mean_list = []
        median_list = []
        max_list = []
        wl = [w]
        for i in range(1, 40):
            print("we're at frame", i)
            abs = np.average(X[:, 0])
            ord = np.average(X[:, 1])
            mean_list.append((abs, ord))
            abs1 = X[np.argmax(w), 0]
            ord1 = X[np.argmax(w), 0]
            max_list.append((abs1, ord1))
            abs2 = ws.weighted_median(X[:, 0], weights=w)
            ord2 = ws.weighted_median(X[:, 1], weights=w)
            median_list.append((abs2, ord2))
            mean_list.append(self.distance(self.calculate_histogram(self.read_image(i)[0],
                                                                     [abs, ord, self.selecz[2], self.selecz[3]], self.Nb)[2]))
            max_list.append(self.distance(self.calculate_histogram(self.read_image(i)[0],
                                                                    [abs1, ord1, self.selecz[2], self.selecz[3]], self.Nb)[2]))
            median_list.append(self.distance(self.calculate_histogram(self.read_image(i)[0],
                                                                       [abs2, ord2, self.selecz[2], self.selecz[3]], self.Nb)[2]))
            for j in range(self.N):
                if w[j] > 0.001:
                    rect = ptch.Rectangle((X[j, 0], X[j, 1]), self.selecz[2], self.selecz[3], linewidth=3, edgecolor='red',
                                          facecolor='None')
                    plt.imshow(self.read_image(i - 1)[0])
                    current_axis = plt.gca()
                    current_axis.add_patch(rect)
                    plt.text(X[j, 0], X[j, 1], f'{w[j]:.3f}', ha='center', va='center', color='white')
            rect2 = ptch.Rectangle((abs2, ord2), self.selecz[2], self.selecz[3], linewidth=3, edgecolor='green',
                                   facecolor='None')
            rect1 = ptch.Rectangle((abs1, ord1), self.selecz[2], self.selecz[3], linewidth=3, edgecolor='blue', facecolor='None')
            rect = ptch.Rectangle((abs, ord), self.selecz[2], self.selecz[3], linewidth=3, edgecolor='red', facecolor='None')
            plt.imshow(self.read_image(i - 1)[0])
            current_axis = plt.gca()
            current_axis.add_patch(rect)
            current_axis.add_patch(rect2)
            current_axis.add_patch(rect1)
            plt.savefig(self.save + '/frame%d.jpg' % i)
            plt.clf()
            A = np.random.choice(range(self.N), self.N, p=w)
            reech = X[A]
            U = np.random.normal(0, 50, (self.N, 2))
            X = reech + U
            print(X)
            w = self.weight(X, i)
            X_l.append(X)
            w = np.array(w)
            w = w / w.sum()
            w = w.flatten()
            wl.append(w)


# User Interface
def main():
    video_path = input("Enter the path of the video: ")
    N = int(input("Enter the value of N the number of particles: "))
    Nb = int(input("Enter the value of Nb the number of clusters to look for: "))
    landa = int(input("Enter the value of landa: "))
    output_path = input("Enter the path of the output: ")

    face_tracker = FaceTracker(input_path=video_path, N=N, Nb=Nb, landa=landa, output_path=output_path)
    face_tracker.track_face()

if __name__ == "__main__":
    main()
