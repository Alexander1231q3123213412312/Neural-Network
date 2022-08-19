import os
from PIL import Image 


strached_images=os.listdir("Russian_tanks_test_source/")

cur=0

images_size=(256,256)

times=6
angle=360.0/times

for image in strached_images:
    img=Image.open('Russian_tanks_test_source/'+image).convert("L").resize(images_size)
    img.save("Russian_tanks_test_L"+str(cur)+".jpg")

    original=img

    for i in range(1,times+1):
        img=original.rotate(angle*i)
        img.save("Russian_tanks_test_L"+str(cur)+".jpg")
        cur+=1
