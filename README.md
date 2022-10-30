# Ground_Estimation
The basic idea of this algorithm is to first use the K-means algorithm to divide the point cloud point set in the .pcd file into multiple clusters, 
and in each cluster,
select the point set with the lowest terrain (z lowest) and the total number of points greater than 10% as the initial ground point set, 
and then use the Ransac algorithm to simulate a plane in this point set, and the points covered by this plane are our selected ground points.
If we want to generate the ground grid S-grid, 
we take the minimum and maximum x-value of the whole point cloud point set, 
generate the x-coordinate of a segmentation point every 1m, and the minimum and maximum y-value, 
generate the y-coordinate of a segmentation point every 1m, and combine the generated x-coordinates and y-coordinates in a complete arrangement 
(i.e., Cartesian product) to get the x, y-coordinates of all vertices of the ground grid. 
After that, the KNN model is used to input the x,y coordinates of the ground grid after the Ransac-generated ground point set is used as the training set, 
and the z coordinates of the ground grid vertices are obtained. The x, y, and z coordinates of the ground grid vertices are stitched together to obtain 
all the vertex coordinates of the ground grid.

Parameter: 
pcd:point cloud data set, input to be transformed into a numpy array, i.e. a 3*n two-dimensional array.
N_CLUSERS: the parameter that controls how many classes Kmeans wants to divide the point cloud into, default is 8
K_SEARCH: originally ransac by pcl, which is a parameter in the pcl library, now run by o3d, the parameter can be ignored.
DISTANCE_THRESHOLD: parameter of Ransac, similar to the hyper plane in SVM, adjust the error range, so that the distance fitted to the plane within the parameter is also considered to belong to the plane of the points
NORMAL_DISTANCE_WEIGHT: parameter of pcl library, can be ignored.
MAX_ITER: control the number of iterations when fitting the ransac plane
K:  KNN model, controls how many proximity points the KNN model uses as reference index, default is 10.


