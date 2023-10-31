import cv2
import numpy as np
import matplotlib.pyplot as plt

#𝑠=𝑐𝑟^𝛾

def kuvvet_donusumu(r, c, gamma):
    r = r.astype(np.float64)#resmi float boyutuna dönüştürmek gerekiyor
    s = c * r ** gamma #üstü yapmak için iki kere çarparız bu da boyutu arttırır 
    s = rescale(s)#değerleri küçültmek için bu fonksiyona ihtiyaç var 
    return s



def rescale(img):#değerler çok büyük hale geleceği için 0-x arasına getirmeliyiz
    img -= np.min(img) # 2-7 -- > 0 - 5  
    img /= np.max(img) # 0 - 1 arasında değerler elde edeceğiz
    img *=255 # 0 - 255 arasına değerlerimizi getirdik
    return img.astype(np.uint8)


mr_path = "./img/mr.tif"
mr_img = cv2.imread(mr_path, 0)

c=1
mr_gammas = [0.6, 0.4, 0.3]
mr_images = []#okunan resimler

for mr_gamma in mr_gammas: #önce 0.6 sonra 0.4 
    donusen = kuvvet_donusumu(mr_img, c, mr_gamma)
    mr_images.append(donusen) 

mr_hstacked_ust = np.hstack((mr_img, mr_images[0]))
mr_hstacked_alt = np.hstack((mr_images[1], mr_images[2]))
mr_vstacked = np.vstack((mr_hstacked_ust, mr_hstacked_alt))

plt.imshow(mr_vstacked, cmap="gray")
plt.show()

sehir_path = "./img/sehir.tif"
sehir_img = cv2.imread(sehir_path, 0)

sehir_gammas = [3, 4, 5]#gama değerleri 3 4 5 
sehir_images = []

for sehir_gamma in sehir_gammas:
    donusen = kuvvet_donusumu(sehir_img,c,sehir_gamma)
    sehir_images.append(donusen)

sehir_hstacked1 = np.hstack((sehir_img, sehir_images[0]))
sehir_hstacked2 = np.hstack((sehir_images[1], sehir_images[2]))
sehir_vstacked = np.vstack((sehir_hstacked1, sehir_hstacked2))

plt.imshow(sehir_vstacked, cmap="gray")
plt.show()