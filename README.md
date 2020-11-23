# Learning to Cluster Faces by Infomap

## Infomap Intorduction
[Infomap Website](https://www.mapequation.org/publications.html#Rosvall-Axelsson-Bergstrom-2009-Map-equation)

## Requirements
* Python >= 3.6
* sklearn
* infomap
* numpy

## Datasets
MS-Celeb-1M : part1_test (584K)
[download](https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md)

## Run
```bash
python face-cluster-by-infomap
```

## Results on part1_test (584K)
| Method | Precision | Recall | F-score |
| ------ |:---------:|:------:|:-------:|
| Chinese Whispers (k=80, th=0.6, iters=20) | 55.49 | 52.46 | 53.93 |
| Approx Rank Order (k=80, th=0) | 99.77 | 7.2 | 13.42 |
| MiniBatchKmeans (ncluster=5000, bs=100) | 45.48 | 80.98 | 58.25 |
| KNN DBSCAN (k=80, th=0.7, eps=0.25, min=1) | 95.25 | 52.79 | 67.93 |
| FastHAC (dist=0.72, single) | 92.07 | 57.28 | 70.63 |
| [DaskSpectral](https://ml.dask.org/clustering.html#spectral-clustering) (ncluster=8573, affinity='rbf') | 78.75 | 66.59 | 72.16 |
| [CDP](https://github.com/XiaohangZhan/cdp) (single model, th=0.7)  | 80.19 | 70.47 | 75.02 |
| [L-GCN](https://github.com/yl-1993/learn-to-cluster/tree/master/lgcn) (k_at_hop=[200, 10], active_conn=10, step=0.6, maxsz=300)  | 74.38 | 83.51 | 78.68 |
| GCN-D (2 prpsls) | 95.41 | 67.77 | 79.25 |
| GCN-D (5 prpsls) | 94.62 | 72.59 | 82.15 |
| GCN-D (8 prpsls) | 94.23 | 79.69 | 86.35 |
| GCN-D (20 prplss) | 94.54 | 81.62 | 87.61 |
| GCN-D + GCN-S (2 prpsls) | 99.07 | 67.22 | 80.1 |
| GCN-D + GCN-S (5 prpsls) | 98.84 | 72.01 | 83.31 |
| GCN-D + GCN-S (8 prpsls) | 97.93 | 78.98 | 87.44 |
| GCN-D + GCN-S (20 prpsls) | 97.91 | 80.86 | 88.57 |
| GCN-V | 92.45 | 82.42 | 87.14 |
| GCN-V + GCN-E | 92.56 | 83.74 | 87.93 |
| Infomap(ours) | 95.50 | 92.51 | 93.98 |

![avatar](./image/evaluate.png)

## References
* [最小熵原理（五）：“层层递进”之社区发现与聚类](https://spaces.ac.cn/archives/7006)
* [人脸聚档主流方案](https://github.com/yl-1993/learn-to-cluster)
