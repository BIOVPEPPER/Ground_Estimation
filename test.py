from functools import lru_cache
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import urllib.request
from pydantic import BaseModel
import numpy as np
import requests
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
import requests
from itertools import product


class Ground_Estimation():
    def __init__(self, pcd, N_CLUSTERS=8, K_SEARCH=50, DISTANCE_THRESHOLD=0.2, NORMAL_DISTANCE_WEIGHT=0.2,
                 MAX_ITER=10000, K=10, use_knn=True, use_kmeans=False):
        if type(pcd) == str:
            print(pcd)
            self.pcd = self.load_pcd_data(pcd)
        elif type(pcd) == type(np.array([0])):
            self.pcd = pcd.astype(np.float32)
        else:
            raise ValueError
        self.N_CLUSTERS = N_CLUSTERS
        self.K_SEARCH = K_SEARCH
        self.DISTANCE_THRESHOLD = DISTANCE_THRESHOLD
        self.NORMAL_DISTANCE_WEIGHT = NORMAL_DISTANCE_WEIGHT
        self.MAX_ITER = MAX_ITER
        self.K = K
        '''
        if use_kmeans:#控制K-means是否使用
            label_pred, _, label_idx = self.kmeans_cluster(self.pcd, self.N_CLUSTERS)
            pred_idx = np.where(label_pred == label_idx)[0]  # coarse ground points idx of the whole scene, high recall
            self.coarse_ground_points = self.pcd[pred_idx, :3]
        else:
            self.coarse_ground_points = self.pcd[:, :3]
        self.fine_ground_idx, self.plane_coefficients = self.ransac_plane_extraction(self.coarse_ground_points,
                                                                                self.K_SEARCH,
                                                                                self.DISTANCE_THRESHOLD,
                                                                                self.NORMAL_DISTANCE_WEIGHT,
                                                                                self.MAX_ITER)
        self.fine_ground_points = self.coarse_ground_points[self.fine_ground_idx, :3]

        if use_knn:#控制KNN是否使用
            self.knn = self.knn_regression_fit(self.fine_ground_points, self.K)
        '''

    def load_pcd_data(self, pcd_file_path):
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        pcd = np.asarray(pcd.points).astype(np.float32)
        return pcd

    def find_center_n_label(self, point_cloud, label_pred, centroids, N_CLUSTERS):
        """
        Find ground point set label, and ground z axis center.
        """
        min_point_num = point_cloud.shape[0] * (1 / N_CLUSTERS)  # 如果点云的数量大于某个阈值，即总点云数量的10%或者由咱们的分类个数决定，我们就把他设为初步地面点击
        point_count = []
        for i in range(N_CLUSTERS):
            point_count.append(np.where(label_pred == i)[0].shape[0])
        sorted_idx = np.argsort(centroids[:, 0])  # min to max idx
        for i in sorted_idx:
            if point_count[i] >= min_point_num:
                return centroids[i], i  # 返回初步地面点集的质心，以及他在第几类

    def kmeans_cluster(self, point_cloud, N_CLUSTERS=8):
        """
        Kmeans_cluster for coarse ground segmentation.

        inputs:
            point_cloud:
                np array, [x, y, z], shape=(n, 3)
            N_CLUSTERS:
        returns:
            label_pred:
            center:
            label_idx:
        """
        estimator = KMeans(n_clusters=N_CLUSTERS)
        estimator.fit(point_cloud)
        label_pred = estimator.labels_
        centroids = estimator.cluster_centers_
        # inertia = estimator.inertia_
        center, label_idx = self.find_center_n_label(point_cloud, label_pred, centroids, N_CLUSTERS)  # Kmeans分类
        return label_pred, center, label_idx

    def ransac_plane_extraction(self,
                                coarse_ground_points,
                                K_SEARCH=50,
                                DISTANCE_THRESHOLD=0.2,
                                NORMAL_DISTANCE_WEIGHT=0.2,
                                MAX_ITER=10000):
        """
        Ransac_plane_extraction for fine grain ground segmentation and plane estimation.

        inputs:
            coarse_ground_points:
                np array, [x, y, z], shape=(n, 3), coarse ground points segmentation
            K_SEARCH:
            DISTANCE_THRESHOLD:
            NORMAL_DISTANCE_WEIGHT:
            MAX_ITER:
        returns:
            indices:
                np array, idx of fine grain ground point from coarse_ground_points
            model:
                list, [a, b, c, d] of plane ax + by + cz + d = 0
        """
        '''
        p = pcl.PointCloud(coarse_ground_points)
        seg = p.make_segmenter_normals(K_SEARCH)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(DISTANCE_THRESHOLD)#有点类似于svm里的超平面，设一个阈值来判断多少范围内的可以被当做inlier
        seg.set_normal_distance_weight(NORMAL_DISTANCE_WEIGHT)
        seg.set_max_iterations(MAX_ITER)
        indices, model = seg.segment()
        '''
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coarse_ground_points)
        model, indices = point_cloud.segment_plane(distance_threshold=DISTANCE_THRESHOLD,
                                                   ransac_n=10,
                                                   num_iterations=MAX_ITER)

        return indices, model

    def knn_regression_fit(self, fine_ground_points, K=10):
        """
        Fit a knn_regression.

        inputs:
            coarse_ground_points:
                np array, [x, y, z], shape=(n, 3), coarse ground points segmentation
            fine_ground_idx:
                np array, idx of fine grain ground point from coarse_ground_points
            K:
        returns:
            fitted knn
        """
        xy = fine_ground_points[:, :2]
        z = fine_ground_points[:, 2]
        knn = KNeighborsRegressor(K)
        knn.fit(xy, z)  # 在模拟出平面的基础上，训练KNN
        return knn

    def knn_predict(self, input_xy):
        """
        Use fitted knn to predict z of given xy.

        inputs:
            input_xy: array, [x, y], shape=(n, 2)
            knn: fitted knn
        returns:
            predicted z of input x,y
        """
        return self.knn.predict(input_xy)

    def expression_predict(self, input_xy):
        """
        Use estimated plane to predict z of given xy.

        inputs:
            input_xy:
                array, [x, y], shape=(n, 2)
            plane_coefficients:
                list, [a, b, c, d] of plane ax + by + cz + d = 0
        returns:
            predicted z of input x,y
        """
        a, b, c, d = self.plane_coefficients[0], self.plane_coefficients[1], self.plane_coefficients[2], \
                     self.plane_coefficients[3]
        return - a / c * input_xy[:, 0] - b / c * input_xy[:, 1] - d / c

    def main_predict(self,
                     point_cloud,
                     N_CLUSTERS=8,
                     K_SEARCH=50,
                     DISTANCE_THRESHOLD=0.2,
                     NORMAL_DISTANCE_WEIGHT=0.2,
                     MAX_ITER=10000,
                     K=10
                     ):
        """
        kmeans cluster --> ransac plane extraction --> knn fit --> return predict func

        inputs:
            point_cloud:
                np array, [x, y, z], shape=(n, 3)
            N_CLUSTERS:
            K_SEARCH:
            DISTANCE_THRESHOLD:
            NORMAL_DISTANCE_WEIGHT:
            MAX_ITER:
            K:
        returns:
            knn:
                fitted knn
            plane_coefficients:
                list, [a, b, c, d] of plane ax + by + cz + d = 0
            knn_predict:
                func use knn to predict
            expression_predict:
                func use expression to predict
        """
        label_pred, _, label_idx = self.kmeans_cluster(point_cloud, N_CLUSTERS)
        pred_idx = np.where(label_pred == label_idx)[0]  # coarse ground points idx of the whole scene, high recall
        coarse_ground_points = point_cloud[pred_idx, :3].astype(np.float32)
        fine_ground_idx, plane_coefficients = self.ransac_plane_extraction(coarse_ground_points, K_SEARCH,
                                                                           DISTANCE_THRESHOLD, NORMAL_DISTANCE_WEIGHT,
                                                                           MAX_ITER)
        knn = self.knn_regression_fit(coarse_ground_points, fine_ground_idx, K)
        return knn, plane_coefficients, self.knn_predict, self.expression_predict

def test_with_Ransac_with_URL(url, max_iter):
    if type(url) == type('a'):
        file_object = requests.get(url)
        with open('requestpcd.pcd', 'wb') as local_file:
            local_file.write(file_object.content)
        pcd_cloud = o3d.io.read_point_cloud('requestpcd.pcd')

    '''
    with tempfile.NamedTemporaryFile(mode='w+t') as tmp_file:
        pcd_path = tmp_file.name
        url
        headers = {
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
        }
        opener = urllib.request.build_opener()
        opener.addheaders = [headers]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, pcd_path)
        print('pcd saved to', pcd_path)
        tmp_file.seek(0)
        pcd_cloud = o3d.io.read_point_cloud('requestpcd.pcd')
        '''
    ge = Ground_Estimation(np.asarray(pcd_cloud.points)[:, 2:3], use_kmeans=True, MAX_ITER=max_iter)
    label_pred, center, label_index = ge.kmeans_cluster(np.asarray(pcd_cloud.points)[:, 2:3])
    pred_ground_idx = np.where(label_pred == label_index)[0]
    ransac_Start = np.asarray(pcd_cloud.points)[pred_ground_idx, :3].astype(np.float32)
    ransac_Indices, ransac_Model = ge.ransac_plane_extraction(ransac_Start, MAX_ITER=max_iter)
    ground_indexes = pred_ground_idx[ransac_Indices]
    ground_Points = np.asarray(pcd_cloud.points)[pred_ground_idx[ransac_Indices], :3]
    min_x = int(min(ground_Points[:, 0]))
    min_y = int(min(ground_Points[:, 1]))
    max_x = int(max(ground_Points[:, 0]))
    max_y = int(max(ground_Points[:, 1]))
    startx = min_x
    starty = min_y
    corner_X = []
    corner_Y = []
    while startx <= max_x:
        corner_X.append(startx)
        startx += 1
    while starty <= max_y:
        corner_Y.append(starty)
        starty += 1
    corner_List = []
    for i in range(len(corner_X)-1):
        for j in range(len(corner_Y)-1):
            corner_List.append([corner_X[i],corner_Y[j]])
            corner_List.append([corner_X[i],corner_Y[j+1]])
            corner_List.append([corner_X[i+1],corner_Y[j]])
            corner_List.append([corner_X[i+1],corner_Y[j+1]])
    corner_List = np.asarray(corner_List)
    knn = KNeighborsRegressor(n_neighbors = 10)
    knn.fit(ground_Points[:, :2], ground_Points[:, 2])
    predict_Z = knn.predict(corner_List[:, :2])
    return corner_List, predict_Z


def predict_points(URL):
    cornerlist, predict_z = test_with_Ransac_with_URL(URL, max_iter=1000)
    all_Points = []
    for i in range(len(predict_z)):
        all_Points.append({'x': int(cornerlist[i][0]), 'y': int(cornerlist[i][1]), 'z': float(predict_z[i])})
    
    return all_Points
    
print(predict_points("https://stardust-data.oss-cn-hangzhou.aliyuncs.com/Clients/上汽/OD融合点云/Production/20220429试标/pcd/1641518451200.pcd"))