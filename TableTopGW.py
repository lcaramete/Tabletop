#!/usr/bin/env python
# coding: utf-8


import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Program to Convert positive list integers to negative and vice-versa
def Convert(lst):
    return [ -i for i in lst ]

#First we calculate the whole process for two consecutive frames
gray1 = cv2.imread("/.../frame1277.jpg",0)
gray2 = cv2.imread("/.../frame1278.jpg",0)

plt.figure( figsize = (40,40))
plt.imshow(gray1, cmap = plt.get_cmap(name = 'gray'))
plt.grid(b=True,color='red')
plt.xticks(np.arange(0, 2290, 100))
plt.yticks(np.arange(0, 1060, 100))
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.show()
plt.figure( figsize = (40,40))
plt.imshow(gray2, cmap = plt.get_cmap(name = 'gray'))
plt.grid(b=True,color='red')
plt.xticks(np.arange(0, 2290, 100))
plt.yticks(np.arange(0, 1060, 100))
plt.tick_params(axis='x', labelsize=40)
plt.tick_params(axis='y', labelsize=40)
plt.show()


#Here se set the position of the central ball using the above grid system
centerx = 440
centery = 1060


#calculate the sum of pixels in each line
#calculate the difference of each line between two consecutive frames
#diffgray is a list containing the difference in grayscale between two consecutive frames
grayuno = [0]*len(gray1)
grayduo = [0]*len(gray2)
for i in range(len(gray1)):
    grayuno[i] = sum(gray1[i])
    grayduo[i] = sum(gray2[i])
print(grayuno[1] - grayduo[1])

diffgray = [0]*len(gray1)
for i in range(len(gray1)):
    diffgray[i] = grayuno[i] - grayduo[i]


#calculate the sum of pixels in each column
#calculate the difference of each line between two consecutive frames
#diffgray is a list containing the difference in grayscale between two consecutive frames
grayunox = [0]*len(gray1[0])
grayduox = [0]*len(gray2[0])
diffgrayx = [0]*len(gray1[0])

print(gray1[1051][2287])
print(len(gray1[0]))
print(len(gray1))
for i in range(len(gray1)):
    for j in range(len(gray1[0])):
        grayunox[j] = grayunox[j] + gray1[i][j]
        grayduox[j] = grayduox[j] + gray2[i][j]
print(grayunox[1] - grayduox[1])


for i in range(len(gray1[0])):
    diffgrayx[i] = grayunox[i] - grayduox[i]


#calculate the maximum and minumum in diffgray list
#compare the two and take the biggest when droping the minus in the negative one
#do not forget to reverse the sign if the min if the biggest 
maxpoint = 0
positionmaxpoint = 0
if max(diffgray) > Convert([min(diffgray)])[0]:
    maxpoint = max(diffgray)
else:
    maxpoint = min(diffgray)

#search the diffgray list and store in positionmaxpoint the position of the peak
for i in range(len(diffgray)):
    if maxpoint == diffgray[i]:
        positionmaxpoint = i
print(positionmaxpoint)

#calculate the maximum and minumum in diffgrayx list
#compare the two and take the biggest when droping the minus in the negative one
#do not forget to reverse the sign if the min if the biggest 
maxpoint = 0
positionmaxpointx = 0
if max(diffgrayx) > Convert([min(diffgrayx)])[0]:
    maxpoint = max(diffgrayx)
else:
    maxpoint = min(diffgrayx)

#search the diffgray list and store in positionmaxpoint the position of the peak
for i in range(len(diffgrayx)):
    if maxpoint == diffgrayx[i]:
        positionmaxpointx = i
print(positionmaxpointx)


#calculate the distance between the peak (ball) and center of the image using x and y
orbitalseparation = math.sqrt(pow((centerx - positionmaxpoint),2) + pow((centery - positionmaxpointx),2))
print(orbitalseparation)


#Test showing the position of the peak difference between the pixels value, basically the position of the moving ball
#on the x axis

x = []
for i in range(len(gray1[0])):
    x.append(i)
y = diffgrayx
plt.figure( figsize = (20,10))
plt.plot(x, y)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.show()



#Test showing the position of the peak difference between the pixels value, basically the position of the moving ball
#on the y axis

x = []
for i in range(len(gray1)):
    x.append(i)
y = diffgray
plt.figure( figsize = (20,10))
plt.plot(x, y)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.show()


#read files from directory, make them gray and store in a list (grayfigures)
#graynumbers are the numbers of the files
#we can choose any part of the movie and specify the start and end frames for selection
startframe = 3427
endframe = 4010
graynumbers = []
for i in range(startframe, endframe):
    graynumbers.append(i)

grayfigures = [0]*len(graynumbers)

for i in range(len(graynumbers)):
    grayfigures[i] = cv2.imread("/.../frame" + str(graynumbers[i]) + ".jpg",0)

plt.imshow(grayfigures[10], cmap = plt.get_cmap(name = 'gray'))
plt.show()


# produce the list of differences between consecutive frames 
# basically generalizing the steps above to all of the frames
orbitalseparationlist = [0]*len(grayfigures)
grayuno = [0]*len(gray1)
grayduo = [0]*len(gray1)
diffgray = [0]*len(gray1)
###
grayunox = [0]*len(gray1[0])
grayduox = [0]*len(gray2[0])
diffgrayx = [0]*len(gray1[0])
###
centerx = 440
centery = 1060

for k in range(len(grayfigures)-1):
    grayuno = [0]*len(gray1)
    grayduo = [0]*len(gray1)
    diffgray = [0]*len(gray1)
    ###
    grayunox = [0]*len(gray1[0])
    grayduox = [0]*len(gray2[0])
    diffgrayx = [0]*len(gray1[0])
    ###
    gray1 = grayfigures[k]
    gray2 = grayfigures[k+1]
    for i in range(len(gray1)):
        grayuno[i] = sum(gray1[i])
        grayduo[i] = sum(gray2[i])
    for i in range(len(gray1)):
        diffgray[i] = grayuno[i] - grayduo[i]
    ###
    for i in range(len(gray1)):
        for j in range(len(gray1[0])):
            grayunox[j] = grayunox[j] + gray1[i][j]
            grayduox[j] = grayduox[j] + gray2[i][j]
    for i in range(len(gray1[0])):
        diffgrayx[i] = grayunox[i] - grayduox[i]
    ###
    maxpoint = 0
    positionmaxpoint = 0
    if max(diffgray) > Convert([min(diffgray)])[0]:
        maxpoint = max(diffgray)
    else:
        maxpoint = min(diffgray)
    #search the diffgray list and store in positionmaxpoint the position of the peak
    for i in range(len(diffgray)):
        if maxpoint == diffgray[i]:
            positionmaxpoint = i
    ###
    maxpoint = 0
    positionmaxpointx = 0
    if max(diffgrayx) > Convert([min(diffgrayx)])[0]:
        maxpoint = max(diffgrayx)
    else:
        maxpoint = min(diffgrayx)
    #search the diffgray list and store in positionmaxpoint the position of the peak
    for i in range(len(diffgrayx)):
        if maxpoint == diffgrayx[i]:
            positionmaxpointx = i
    #calculate the distance between the peak (ball) amd center of the image
    orbitalseparation = 0
    orbitalseparation = math.sqrt(pow((centerx - positionmaxpoint),2) + pow((centery - positionmaxpointx),2))
    orbitalseparationlist[k] = orbitalseparation



x = []
for i in range(len(orbitalseparationlist)):
    x.append(float(i))
lo = orbitalseparationlist
hplus = [0]*len(orbitalseparationlist)

# total mass and chirp mass of the system in grams
m1 = 100
m2 = 560
M = m1+m2 #total mass
M_chirp = pow(m1*m2,3/5)/pow(M,1/5) # chirp mass

for i in range(len(lo)-1):
    hplus[i] = pow(M_chirp,5/3)*pow(M,1/2)*pow((1/lo[i]),1)*(math.cos( (len(lo)-x[i])**(5/8) ))

print(hplus[0])



fig = plt.figure( figsize = (20,20))
fig.patch.set_facecolor('xkcd:white')

plt.scatter(x[:550], y[:550], color='r')

# naming the x axis
plt.xlabel('t(s)', fontsize=50)
# naming the y axis
plt.ylabel(r'$h_{+} \times 10^{-21}$', fontsize=50)
# giving a title to my graph
# plt.title('Plot', fontsize=20)
# seting tick labels size
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# saving the figure
plt.savefig("/.../waveform", dpi=800)

# function to show the plot
plt.show()

################ Ploting an interpolation

# plotting the points
fig = plt.figure( figsize = (20,20))
fig.patch.set_facecolor('xkcd:white')

plt.scatter(x[:550], y[:550], color='r')    
    
# naming the x axis
plt.xlabel('t(s)', fontsize=50)
# naming the y axis
plt.ylabel(r'$h_{+} \times 10^{-21}$', fontsize=50)
# giving a title to my graph
# plt.title('Plot', fontsize=20)
# seting tick labels size
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

xs = np.linspace(0,583,1000)

spl = UnivariateSpline(x, y)
spl.set_smoothing_factor(20800000)
plt.plot(xs[:930], spl(xs)[:930], 'b', lw=3)

# saving the figure
plt.savefig("/.../waveform_+_interpolation", dpi=800)

# function to show the plot
plt.show()

################

file1 = open("/.../hplus_all.txt","w")
file1.write(str(hplus))
file1.close()



file1 = open("/.../orbitalseparation.txt","w")
file1.write(str(orbitalseparationlist))
file1.close()






