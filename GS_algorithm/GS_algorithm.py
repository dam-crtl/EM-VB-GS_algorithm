import numpy as np
from scipy.stats import multivariate_normal

def gibbs_sampling(data, num_clusters, num_iterations):
    num_points = len(data)
    
    # 初期化: ランダムにクラスターを割り当てる
    assignments = np.random.randint(num_clusters, size=num_points)

    # 各クラスターの初期平均と共分散行列を設定
    means = np.random.randn(4, 3)
    covs = [np.eye(3) for _ in range(num_clusters)]
    #print(means)
    #print(covs)
    # サンプリングループ
    print(means[3])
    for _ in range(num_iterations):
        # 各データポイントについて
        for i in range(num_points):
            point = data[i]

            # 現在のクラスターを取得
            current_cluster = assignments[i]

            # 現在のクラスターからのポイントを削除
            counts = np.bincount(assignments, minlength=num_clusters)
            counts[current_cluster] -= 1
            print(means[3])
            # 各クラスターごとに対数尤度を計算
            log_likelihoods = []
            for j in range(num_clusters):
                print(j)
                print(means[j])
                #print(covs[j])
                log_likelihood = np.log(counts[j] + 1) + multivariate_normal.logpdf(point, means[j], covs[j], allow_singular=True)
                log_likelihoods.append(log_likelihood)

            # クラスターをサンプリング
            probabilities = np.exp(log_likelihoods - np.max(log_likelihoods))
            probabilities /= np.sum(probabilities)
            new_cluster = np.random.choice(num_clusters, p=probabilities)

            # 新しいクラスターにポイントを割り当て
            assignments[i] = new_cluster
            counts[new_cluster] += 1

        # 各クラスターごとに平均と共分散行列を更新
        for j in range(num_clusters):
            points = data[assignments == j]
            #print(j)
            #print(points)
            means[j] = np.mean(points, axis=0)
            covs[j] = np.cov(points.T)

    return assignments, means, covs


# テストデータ生成
np.random.seed(0)
data = np.concatenate([
    np.random.multivariate_normal([1, 1, 1], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], size=100),
    np.random.multivariate_normal([4, 4, 4], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], size=100),
    np.random.multivariate_normal([7, 7, 7], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], size=100),
    np.random.multivariate_normal([10, 10, 10], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], size=100)
])

# クラスタリングの実行
num_clusters = 4
num_iterations = 100
assignments, means, covs = gibbs_sampling(data, 4, num_iterations)
a
# 結果表示
print("クラスタリング結果:")
for i in range(num_clusters):
    cluster_points = data[assignments == i]
    print(f"クラスター {i + 1}: ポイント数 = {len(cluster_points)}, 平均 = {means[i]}, 共分散行列 = \n{covs[i]}")
