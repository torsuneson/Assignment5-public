# Assignment 5

D7015B-IAI-and-eM\
\
Tor Suneson

No report was needed

-> Documtenation on GitHub

Orchestration

```text
process_dataset(npy_path: str, min_samples: int = 5)

for both datasets. 
```

## Task1 ( Grade 3 )

• Find the best value for the ground level\
• One way to do it is using a histogram *np.histogram*\
• Update the function get_ground_level() with your changes

Code section marked with **Task 1**

Functions:

```text
# modyfied 
get_ground_level(pcd, n_bins: int = 256, window_bins: int = 2) -> float

# saves histogram for HTML report 
save_ground_histogram(z_values, z_ground, out_png):
```

For both the datasets

• Report the ground level in the readme file in your github project     \
• Add the histogram plots to your project Readme

### For Dataset 1

**Ground level**: 61.277

![dataset1\_hist\_ground.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset1_hist_ground.png?raw=true)

### For Dataset 2

**Ground level**: 61.237

![dataset2\_hist\_ground.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset2_hist_ground.png?raw=true)

## Task 2 (Grade 4)

• Find an optimized value for eps.

• Plot the elbow and extract the optimal value from the plot

• Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

Code section marked with **Task 2**

Functions

```text
k_distance_curve(points_xy: np.ndarray, k: int)

find_elbow_eps(k_dists: np.ndarray)

save_k_distance_plot(k_dists, eps, knee_idx, out_png)

run_dbscan(points_xy: np.ndarray, eps: float, min_samples: int=5)

save_cluster_scatter_xy(points_xy, labels, title, out_png)
```

For both the datasets

* Report the optimal value of eps in the README to your github projects

* Add the Elbow plots to yout github project README

* Add the cluster pots to yout github projects README

### Dataset 1

**Optimal eps**: 0.6168 (knee idx=58777)

![dataset1\_kdist\_elbow.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset1_kdist_elbow.png?raw=true)![dataset1\_dbscan\_clusters.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset1_dbscan_clusters.png?raw=true)

### Dataset 2

**Optimal eps**: 0.4312 (knee idx=72409)

![dataset2\_kdist\_elbow.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset2_kdist_elbow.png?raw=true)![dataset2\_dbscan\_clusters.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset2_dbscan_clusters.png?raw=true)

## Task 3 ( Grade 5)

* Find the largest cluster, since that should be hte catenary cluster, beware of the noise cluster

* Use hte x,y span for hte culster to find the largest cluster

Functions

```text
get_largest_cluster_by_xy_span(points_xy: np.ndarray, labels: np.ndarray)

save_largest_cluster_plot(points_xy, target_mask, title, out_png)
```

For both databases

* Report min(x), min(y), max(x), max(y) for the cantenary cluster in rhe README of your github project

* Add the plot of the centary cluster to the README

### Dataset 1

**BBox**: min(x)=26.498, min(y)=80.012, max(x)=62.140, max(y)=159.997

![dataset1\_largest\_cluster.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset1_largest_cluster.png?raw=true)

Image not rendered in 3D. For 3D rotation see the html file in[ /images/](https://github.com/torsuneson/Assignment5-public/blob/main/images/report.html)

### Dataset 2

**BBox**: min(x)=11.393, min(y)=0.043, max(x)=37.007, max(y)=79.993

![dataset2\_largest\_cluster.png?raw=true](https://github.com/torsuneson/Assignment5-public/blob/63b82588db46364696ec9caae8f5c55530d84b39/images/dataset2_largest_cluster.png?raw=true)

Image not rendered in 3D. For 3D rotation see the html file in[ /images/](https://github.com/torsuneson/Assignment5-public/blob/main/images/report.html)

Summary HTML stored in /images/ named report.hml
