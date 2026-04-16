from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


# Datasets for contrastive pretraining and repertoire-level classification.

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

    def get_probabilities(self, n, weights):
        if self.use_weight is False:
            return None
        assert len(weights) == n, f"Weight length mismatch: {len(weights)} vs {n}"

        # Validate and normalize sampling weights for one bag.
        weights_arr = np.array(weights, dtype=np.float64)
        negative_indices = np.where(weights_arr < 0)[0]
        if len(negative_indices) > 0:
            print(f"Negative weight values: {weights_arr[negative_indices]}")
            raise ValueError("Weight array contains negative values")

        if np.any(weights_arr < 0):
            raise ValueError("Weights contain negative values")
        if np.any(np.isnan(weights_arr)) or np.any(np.isinf(weights_arr)):
            raise ValueError("Weights contain NaN or infinity")

        # Fall back to uniform sampling when all weights are zero.
        total = np.sum(weights_arr)
        if total == 0:
            return np.ones(n) / n
        else:
            prob = weights_arr / total
            # Guard against tiny numerical drift before final normalization.
            prob = np.clip(prob, 0.0, 1.0)
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
