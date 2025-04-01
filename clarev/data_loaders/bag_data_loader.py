from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


## Unsupervised training for contrast learning

class ClusterBagDataset(Dataset):  ## contains clusters after mapping to centroids

    '''
    superbags: List[List[tensor]]; a list of samples, each sample contains a list of tensors, with the same v gene type order
    '''

    def __init__(self, superbags, weight = None, subbag_size = 30, data_repeat = 1):
        self.bags = []  ## one bag is a tensor matrix
        self.sample_ids = []
        self.bag_orders = []
        self.subbag_size = subbag_size
        self.num_features = 0
        self.sample_num = 0
        self.data_repeat = data_repeat
        self.weights = [] ## weight for each cluster
        self.use_weight = False

        assert isinstance(superbags, list), "superbags should be a 2-dim list"
        assert superbags[0][0].shape[1] is not None, "superbags should be a 2-dim list, contains tensors"

        if weight is not None:
            assert len(weight) == len(superbags), "Weight structure mismatch"
            assert all(len(w) == len(s) for w, s in zip(weight, superbags)), "Weight sublist length mismatch"
            self.use_weight = True
        
        self.split_bags(superbags, weight)

    def split_bags(self, superbags, weight=None):
        cluster_num = len(superbags[0])
        self.num_features = superbags[0][0].shape[1]
        self.sample_num = len(superbags)

        for smp_id in range(self.sample_num):
            self.sample_ids += [smp_id] * cluster_num
            self.bag_orders += [i for i in range(cluster_num)]
            self.bags += superbags[smp_id]  ## a list of tensors

            if weight is not None:
                self.weights += weight[smp_id] ## flatten the weight list
            else:
                self.weights += [None] * cluster_num

    # def get_probabilities(self, n, weights):
    #     if self.use_weight is False:
    #         return None
    #     assert len(weights) == n, f"Weight length mismatch: {len(weights)} vs {n}"
    #     return np.array(weights) / np.sum(weights)  # norm within the bag
        

    def get_probabilities(self, n, weights):
        if self.use_weight is False:
            return None
        assert len(weights) == n, f"权重长度不匹配: {len(weights)} vs {n}"
        
        # 转换权重为数组并检查非负
        weights_arr = np.array(weights, dtype=np.float64)
        # 检查负值
        negative_indices = np.where(weights_arr < 0)[0]
        if len(negative_indices) > 0:
            print(f"对应权重值: {weights_arr[negative_indices]}")
            raise ValueError("权重数组包含负值")

        if np.any(weights_arr < 0):
            raise ValueError("权重包含负值")
        if np.any(np.isnan(weights_arr)) or np.any(np.isinf(weights_arr)):
            raise ValueError("权重包含NaN或无限值")
        
        # 处理所有权重为零的情况
        total = np.sum(weights_arr)
        if total == 0:
            # 退化为均匀分布
            return np.ones(n) / n
        else:
            prob = weights_arr / total
            # 处理可能的浮点误差
            prob = np.clip(prob, 0.0, 1.0)
            # 最后再次归一化确保总和为1
            prob /= prob.sum()
            return prob
    def __len__(self):
        return len(self.bags) * self.data_repeat

    def __getitem__(self, idx):


        def weighted_choice(size, population, weights):
                """
                size = number of TCRs in a subbag (user setted sample num)
                population = number of TCRs in original bag
                weights = weight for each TCR in the original bag
                """
                
                prob = self.get_probabilities(population, weights) ## get the probability for each TCR in the bag

                return np.random.choice(
                    population,  # size of pool
                    size=size,   # number of samples
                    replace=True,
                    p=prob   ## probability for each TCR
                )
    
        idx = idx % len(self.bags)

        bag1 = self.bags[idx]
        sample_id = self.sample_ids[idx]
        bag_order = self.bag_orders[idx]

        weights1 = self.weights[idx] ## weight for bag1


        ## sample a negative bag with the same bag_order and different sample_id

        neg_sample_id = np.random.choice([i for i in range(self.sample_num) if i != sample_id])
        neg_idx = self.sample_ids.index(neg_sample_id) + bag_order
        bag2 = self.bags[neg_idx]  ## negative bag
        weights2 = self.weights[neg_idx] ## weight for bag2

        ## sample subbags
        pos_sample1_padded = np.zeros((self.subbag_size, self.num_features))
        pos_sample2_padded = np.zeros((self.subbag_size, self.num_features))
        neg_sample_padded = np.zeros((self.subbag_size, self.num_features))

        # pos_idx1 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_idx1 = weighted_choice(
            self.subbag_size, 
            len(bag1), 
            weights1
        )
        pos_sample1_padded[:len(pos_idx1)] = bag1[pos_idx1]

        # pos_idx2 = np.random.choice(len(bag1), self.subbag_size, replace=True)
        pos_idx2 = weighted_choice(
            self.subbag_size, 
            len(bag1), 
            weights1
        )
        pos_sample2_padded[:len(pos_idx2)] = bag1[pos_idx2]
        

        # neg_idx = np.random.choice(len(bag2), self.subbag_size, replace=True)  ## sample M instances from bag2
        neg_idx = weighted_choice(
            self.subbag_size, 
            len(bag2), 
            weights2
        )
        neg_sample_padded[:len(neg_idx)] = bag2[neg_idx]
        return pos_sample1_padded, pos_sample2_padded, neg_sample_padded
    
# class ClusterBagDataset(Dataset):  ## contains clusters after mapping to centroids

#     '''
#     superbags: List[List[tensor]]; a list of samples, each sample contains a list of tensors, with the same v gene type order
#     '''

#     def __init__(self, superbags, use_simu_data = False, subbag_size = 30, data_repeat = 1):
#         self.bags = []  ## one bag is a tensor matrix
#         self.sample_ids = []
#         self.bag_orders = []
#         self.subbag_size = subbag_size
#         self.num_features = 0
#         self.sample_num = 0
#         self.data_repeat = data_repeat

#         if use_simu_data:
#             self.simu_data()
#         else:
#             ## check if features is a dict
#             assert isinstance(superbags, list), "superbags should be a 2-dim list"
#             assert superbags[0][0].shape[1] is not None, "superbags should be a 2-dim list, contains tensors"
#             self.split_bags(superbags)

#     def split_bags(self, superbags):
#         cluster_num = len(superbags[0])
#         self.num_features = superbags[0][0].shape[1]
#         self.sample_num = len(superbags)

#         for smp_id in range(self.sample_num):
#             self.sample_ids += [smp_id] * cluster_num
#             self.bag_orders += [i for i in range(cluster_num)]
#             self.bags += superbags[smp_id]  ## a list of tensors

#     def simu_data(self):
#         num_samples = 50
#         num_bags = 10
#         min_instances = 20
#         max_instances = 100
#         num_features = 96

#         self.sample_num = num_samples

#         print("Simulate data for testing ...")

#         for smp_id in range(num_samples):
#             X, y = make_blobs(n_samples=max_instances * num_bags, centers=1, n_features=num_features)
#             gmm = GaussianMixture(n_components=3)
#             gmm.fit(X)

#             for bag_order in range(num_bags):
#                 num_instances = np.random.randint(min_instances, max_instances)
#                 X_new, _ = gmm.sample(num_instances)
#                 X_new = X_new + bag_order * 0.1
#                 self.bags.append(torch.from_numpy(X_new))
#             self.sample_ids += [smp_id] * num_bags
#         self.bag_orders = [i for i in range(num_bags)] * num_samples
#         self.num_features = num_features

#         print("Done. ({} bags, {} samples, {}-dim features)".format(len(self.bags), self.sample_num, self.num_features))
#         # ### show the data
#         # print("Sample 1 bag 0: ", self.bags[0].shape)
#         # print("bag 0's feature: ", self.bags[0])
#         # print("Sample 1 , last bag: ", self.bags[num_bags - 1].shape)
#         # print("last bag's feature: ", self.bags[num_bags - 1])
#         # print("size of self.bags: ", len(self.bags))
#         # print("size of self.sample_ids: ", len(self.sample_ids))
#         # print("size of self.bag_orders: ", len(self.bag_orders))

#     def __len__(self):
#         return len(self.bags) * self.data_repeat

#     def __getitem__(self, idx):

#         idx = idx % len(self.bags)

#         bag1 = self.bags[idx]
#         sample_id = self.sample_ids[idx]
#         bag_order = self.bag_orders[idx]

#         ## sample a negative bag with the same bag_order and different sample_id

#         neg_sample_id = np.random.choice([i for i in range(self.sample_num) if i != sample_id])

#         neg_idx = self.sample_ids.index(neg_sample_id) + bag_order
#         bag2 = self.bags[neg_idx]  ## negative bag

#         ## sample subbags
#         pos_sample1_padded = np.zeros((self.subbag_size, self.num_features))
#         pos_sample2_padded = np.zeros((self.subbag_size, self.num_features))
#         neg_sample_padded = np.zeros((self.subbag_size, self.num_features))

#         pos_idx1 = np.random.choice(len(bag1), self.subbag_size, replace=True)
#         pos_sample1_padded[:len(pos_idx1)] = bag1[pos_idx1]

#         pos_idx2 = np.random.choice(len(bag1), self.subbag_size, replace=True)
#         pos_sample2_padded[:len(pos_idx2)] = bag1[pos_idx2]
        
#         neg_idx = np.random.choice(len(bag2), self.subbag_size, replace=True)  ## sample M instances from bag2
#         neg_sample_padded[:len(neg_idx)] = bag2[neg_idx]

#         return pos_sample1_padded, pos_sample2_padded, neg_sample_padded



# class RepertoireDataset(Dataset):
#     '''
#     superbags: List[List[tensor]]; a list of samples, each sample contains a list of tensors, with the same v gene type order
#     return a list of repertoire, which inclued sampled bags, use np.random.choice to sample bags
#     '''

#     def __init__(self, superbags, labels, v_freq_mtx, subbag_size = 30):
#         self.superbags = superbags
#         self.subbag_size = subbag_size
#         self.labels = labels
#         self.v_freq_mtx = v_freq_mtx

#         assert len(self.superbags) == len(self.labels), " num of labels should be the same as num of samples"
    
#     def __len__(self):
#         return len(self.superbags)
    
#     def __getitem__(self, idx):
#         repertoire = self.superbags[idx]
#         sampled_bags = np.zeros((len(repertoire), self.subbag_size, repertoire[0].shape[1]))
#         for vidx, vbag in enumerate(repertoire):
#             selected_idx = np.random.choice(len(vbag), self.subbag_size, replace=True)
#             sampled_bags[vidx] = vbag[selected_idx]
        
#         return torch.tensor(sampled_bags, dtype=torch.float32), self.labels[idx], torch.tensor(self.v_freq_mtx[idx], dtype=torch.float32)  # shape: (num_v_genes, subbag_size, feature_dim), shape: (1,), shape: (num_v_genes,)



class RepertoireDataset(Dataset):
    '''
    superbags: List[List[tensor]]; a list of samples, each sample contains a list of tensors, with the same v gene type order
    return a list of repertoire, which inclued sampled bags, use np.random.choice to sample bags
    '''

    def __init__(self, superbags, labels, v_freq_mtx, weight=None, subbag_size = 30):
        self.superbags = superbags
        self.subbag_size = subbag_size
        self.labels = labels
        self.v_freq_mtx = v_freq_mtx
        self.weights = weight
        if self.weights is not None:
            self.use_weight = True
        else:
            self.use_weight = False

        assert len(self.superbags) == len(self.labels), " num of labels should be the same as num of samples"
    
    def __len__(self):
        return len(self.superbags)

    def get_probabilities(self, n, weights):
        if self.use_weight is False:
            return None
        assert len(weights) == n, f"length of weights not match with num of TCRs{len(weights)} vs {n}"
        
        weights_arr = np.array(weights, dtype=np.float64)
        total = np.sum(weights_arr)
        if total == 0:
            return np.ones(n) / n
        else:
            prob = weights_arr / total
            prob = np.clip(prob, 0.0, 1.0)
            prob /= prob.sum()
            return prob

    def __getitem__(self, idx):

        def weighted_choice(size, population, weights):
            """
            size = number of TCRs in a subbag (user setted sample num)
            population = number of TCRs in original bag
            weights = weight for each TCR in the original bag
            """
            
            prob = self.get_probabilities(population, weights) ## get the probability for each TCR in the bag

            return np.random.choice(
                population,  # size of pool
                size=size,   # number of samples
                replace=True,
                p=prob   ## probability for each TCR
            )

        repertoire = self.superbags[idx]
        sampled_bags = np.zeros((len(repertoire), self.subbag_size, repertoire[0].shape[1]))
        for vidx, vbag in enumerate(repertoire):

            weights_in_bag = self.weights[idx][vidx]
            if self.weights is False:
                selected_idx = np.random.choice(len(vbag), self.subbag_size, replace=True)
            else:
                selected_idx = weighted_choice(
                            self.subbag_size, 
                            len(vbag), 
                            weights_in_bag
                            )
            sampled_bags[vidx] = vbag[selected_idx]
        
        return torch.tensor(sampled_bags, dtype=torch.float32), self.labels[idx], torch.tensor(self.v_freq_mtx[idx], dtype=torch.float32)  # shape: (num_v_genes, subbag_size, feature_dim), shape: (1,), shape: (num_v_genes,)

class RepertoireAggregateDataset(Dataset):
    '''
    superbags: List[List[tensor]]; a list of samples, each sample contains a list of tensors, with the same v gene type order
    return a list of repertoire, which inclued tensor of aggregated features for each repertoire
    '''
    def __init__(self, superbags, labels, v_freq_mtx, agg_type = "mean"):
        self.superbags = superbags
        self.labels = labels
        self.v_freq_mtx = v_freq_mtx
        self.agg_type = agg_type
        self.emb_dim = superbags[0][0].shape[1] if len(superbags[0][0]) > 0 else 120

        assert len(self.superbags) == len(self.labels), " num of labels should be the same as num of samples"
    
    def __len__(self):
        return len(self.superbags)

    def aggregate_bag_feature(self, repertoirebag):
        '''
        repertoirebags: list of tensor, each tensor is a bag of embeddings
        agg_type: str, "mean" or "max"

        return: tensor, aggregated features for each repertoire (v_num * emb_dim)
        '''
        agg_features = torch.zeros(len(repertoirebag), self.emb_dim)  ## v_num * emb_dim

        if self.agg_type == "mean":
            for i, bag in enumerate(repertoirebag):
                bag = torch.tensor(bag, dtype=torch.float32)
                agg_features[i] = torch.mean(bag, dim=0)

        elif self.agg_type == "max":
            for i, bag in enumerate(repertoirebag):
                bag = torch.tensor(bag, dtype=torch.float32)
                agg_features[i] = torch.max(bag, dim=0).values

        return agg_features
    
    def __getitem__(self, idx):
        repertoire = self.superbags[idx]
        agg_features = self.aggregate_bag_feature(repertoire)
        return agg_features, self.labels[idx], torch.tensor(self.v_freq_mtx[idx], dtype=torch.float32)


if __name__ == "__main__":

    ## use one dataset
    dataset = ClusterBagDataset(superbags=None, use_simu_data = True)
    dataset.__getitem__(105)
    print("dataset length: ", len(dataset))
    print("dataset num_features: ", dataset.num_features)
    print("example bag length = ", len(dataset.__getitem__(105))) ## tuple of 3 tensors
    print("example pos_smp shape = ", dataset.__getitem__(105)[0].shape) ## (subbag_size, num_features)




