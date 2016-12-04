import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.data import binary_blobs
import cv2
import math

######################################################################
# **Medial axis skeletonization**
#
# The medial axis of an object is the set of all points having more than one
# closest point on the object's boundary. It is often called the *topological
# skeleton*, because it is a 1-pixel wide skeleton of the object, with the same
# connectivity as the original object.
#
# Here, we use the medial axis transform to compute the width of the foreground
# objects. As the function ``medial_axis`` returns the distance transform in
# addition to the medial axis (with the keyword argument ``return_distance=True``),
# it is possible to compute the distance to the background for all points of
# the medial axis with this function. This gives an estimate of the local width
# of the objects.
#
# For a skeleton with fewer branches, ``skeletonize`` or ``skeletonize_3d``
# must be preferred.

from skimage.morphology import medial_axis, skeletonize, skeletonize_3d
from skimage.filters import threshold_otsu
from scipy import ndimage
import numpy as np

# Generate the data
#data = binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)
#data = cv2.imread('C:\Users\Peter\Downloads\cracks.jpg', 1)
image = cv2.imread('C:\Users\Peter\Desktop\square.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
#data = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ret, data = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('pic', data)
#cv2.waitKey(0)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img, contours, -1, (127,127,0), 3)
#edges = cv2.Canny(img, 150, 300, 25)
#cv2.imshow('edges', edges)
thresh = threshold_otsu(image)
data = image < thresh
#n=[[0,1,0],[1,1,1],[0,1,0]];
n=[[1,1,1],[1,1,1]];
data = ndimage.binary_erosion(data, structure = np.ones((3,3))).astype(data.dtype)
#data = ndimage.binary_erosion(data, structure = n).astype(data.dtype)
'''open = ndimage.binary_opening(data)
eroded = ndimage.binary_erosion(data)
reconstruction = ndimage.binary_propagation(eroded, mask=data)
data=reconstruction'''
'''sift = xfeatures2d.SIFT_create()
kp=sift.detect(data, None)
img=cv2.drawKeypoints(data,kp)'''
#cv2.imwrite('sift', img)
# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)

# Compare with other skeletonization algorithms
skeleton = skeletonize(data)
skeleton3d = skeletonize_3d(data)
pixels = np.where(skeleton3d == 255)
i = 0;
p = [];
print pixels[0].size
while i < pixels[0].size:
    p = p + [[pixels[0][i], pixels[1][i]]];
    i=i+1;
#print p
#print len(p)

n=0;
points=p;
start = [points[0][0], points[0][1]];
path=[start];
current_p = [points[0][0], points[0][1]];
#print current_p
del points[0]
while len(points)>0:
    i=0;
    d =[];
    while i<len(points):
        new_p=points[i];
        #print new_p
        #print current_p
        dy = new_p[1]-current_p[1];
        dx = new_p[0]-current_p[0];
        d = d+[math.sqrt(dx**2+dy**2)];
        #start=current_p;
        i = i+1;
    #print d
    #print d.index(min(d))
    path=path+[points[d.index(min(d))]];
    current_p = points[d.index(min(d))];
    #print current_p
    del points[d.index(min(d))];
path=path+[start];
print path

i=0;
slope=[];
while i<pixels[0].size-1:
    dy=pixels[1][i+1]-pixels[1][i];
    dx=pixels[0][i+1]-pixels[0][i];
    if dx==0:
        slope=slope+['vertical']
    else:
        slope=slope+[dy/dx];
    i=i+1;
#print slope
#print pixels[0][0]
#print pixels[1][0]
#print pixels
# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

ax[0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
ax[1].contour(data, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')

ax[2].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('skeletonize')
ax[2].axis('off')

ax[3].imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('skeletonize_3d')
ax[3].axis('off')

fig.tight_layout()
plt.show()
