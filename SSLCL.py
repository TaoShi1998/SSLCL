import torch
import torch.nn as nn
from torch.nn.functional import normalize




class SSLCL(nn.Module):
    '''
    pos_focus_param: positive focusing parameter, which controls the strength of penalties on hard positive samples; 
    neg_focus_param: negative focusing parameter, which controls the strength of penalties on hard negative samples;
    sample_weight_param: sample weight parameter, which assigns higher weights to minority emotions;
    pos_aug_param: positive pairs augmentation parameter, it is a boolean variable, if set to True, multimodal views will be utilized as 
    data augmentations to generate additional positive sample-label pairs;
    label_loss_param: label-label loss parameter, which controls the contributions from label-label discrimination loss;
    num_classes: 7 for MELD and 6 for IEMOCAP;
    device: cpu or gpu.
    '''
    def __init__(self, pos_focus_param, neg_focus_param, sample_weight_param, pos_aug_param, label_loss_param, num_classes, device):
        super().__init__()

        self.pos_focus_param = pos_focus_param
        self.neg_focus_param = neg_focus_param
        self.sample_weight_param = sample_weight_param
        self.pos_aug_param = pos_aug_param
        self.label_loss_param = label_loss_param
        self.num_classes = num_classes
        self.device = device   

        self.label_indices = torch.LongTensor([i for i in range(self.num_classes)]).to(self.device)
    

    '''
    Measure the correlation between two random variables X and Y using Soft-HGR maximum correlation. 
    '''
    def soft_HGR_correlation(self, X_embs, Y_embs):
        X_embs_mean = torch.mean(X_embs, dim = 0)
        Y_embs_mean = torch.mean(Y_embs, dim = 0)
        zero_mean_X_embs = X_embs - X_embs_mean
        zero_mean_Y_embs = Y_embs - Y_embs_mean
        X_Y_embs_expe = torch.sum(zero_mean_X_embs * zero_mean_Y_embs, dim = -1) / (zero_mean_X_embs.shape[0] - 1)
        X_embs_corr = torch.cov(zero_mean_X_embs)
        Y_embs_corr = torch.cov(zero_mean_Y_embs)
        X_Y_embs_corr = torch.diagonal(X_embs_corr @ Y_embs_corr)
        corr = X_Y_embs_expe - X_Y_embs_corr / 2

        return corr
    

    '''
    Measure the similarity between different label embeddings through dot products.
    '''
    def dot_product_similarity(self, label_X_embs, label_Y_embs):
        similarity = torch.sum(label_X_embs * label_Y_embs, dim = -1)

        return similarity

    
    '''
    Convert correlations to probabilities using the Softmax function. 
    '''
    def corr_to_prob(self, corr):
        prob = torch.softmax(corr, dim = 0)

        return prob


    '''
    Calculate the loss from a positive sample-label pair. sample feature i and label embedding j is 
    considered as a positive sample-label pair if j is the corresponding ground-truth label for i.
    '''
    def positive_pairs_loss(self, pos_prob):
        pos_loss = torch.log(pos_prob) * (1 - pos_prob)**self.pos_focus_param

        return pos_loss
    

    '''
    Calculate the loss from a negative sample-label pair. sample feature i and label embedding 
    j is considered as a negative sample-label pair if j is not the corresponding ground-truth label for i.
    '''
    def negative_pairs_loss(self, neg_prob):
        neg_loss = torch.log(1.0 - neg_prob) * neg_prob**self.neg_focus_param

        return neg_loss
    

    '''
    Assign more weight to minority samples. 
    '''
    def sample_weight(self):
        counted_labels = self.labels.unique(return_counts = True)
        num_available_classes = len(counted_labels[0])
        class_weights = torch.zeros(self.num_classes).to(self.device)
        for i in range(num_available_classes):
            class_weights[counted_labels[0][i]] = (self.num_samples / counted_labels[1][i])**self.sample_weight_param
        class_weights /= torch.min(class_weights[class_weights != float(0)])
        batch_sample_weights = torch.tensor([class_weights[label] for label in self.labels]).to(self.device)

        return batch_sample_weights


    '''
    Calculate the total loss in the batch:
    f_tav_embs: multimodal sample features f used as positive sample features. f_tav_embs.shape: (num_samples, num_features);
    f_t_embs: the textual modality of sample features f used for positive pairs augmentation. f_t_embs.shape: (num_samples, num_features);
    f_ta_embs: the textual + audio modality of sample features f used for positive pairs augmentation. f_ta_embs.shape: (num_samples, num_features);
    f_tv_embs: the textual + visual modality of sample features f used for positive pairs augmentation. f_tv_embs.shape: (num_samples, num_features);
    label_embs: the set of learned label embeddings. label_embs.shape: (num_classes, num_features);
    labels: the corresponding ground-truth discrete emotion labels in this batch. labels.shape: (num_samples, 1).
    '''
    def forward(self, f_t_embs, f_ta_embs, f_tv_embs, f_tav_embs, label_embs, labels):
        f_t_embs = normalize(f_t_embs, dim = -1)
        label_embs = normalize(label_embs, dim = -1)
        
        if self.pos_aug_param == True:
            f_ta_embs = normalize(f_ta_embs, dim = -1)
            f_tv_embs = normalize(f_tv_embs, dim = -1)
            f_tav_embs = normalize(f_tav_embs, dim = -1)

        self.labels = labels
        self.num_samples = f_t_embs.shape[0]
        self.feature_dim = f_t_embs.shape[-1]

        batch_label_embs = torch.stack(([label_embs[label] for label in labels]))

        all_labels = self.label_indices.expand_as(torch.zeros(self.num_samples, self.num_classes)).to(self.device).transpose(1, 0).clone()
        all_labels[0] = labels
        for i in range(self.num_samples):
            all_labels[labels[i]][i] = 0
        all_neg_labels = all_labels[1:]

        idx = torch.randperm(all_neg_labels.shape[0])
        shuffled_all_neg_labels = all_neg_labels[idx]

        feature_label_corrs_list = []

        if self.pos_aug_param == True:
            for f_embs in [f_t_embs, f_ta_embs, f_tv_embs, f_tav_embs]:
                pos_feature_label_corr = self.soft_HGR_correlation(f_embs, batch_label_embs)
                feature_label_corrs_list.append(pos_feature_label_corr)
            for neg_labels in shuffled_all_neg_labels:
                neg_label_embs = torch.stack(([label_embs[neg_label] for neg_label in neg_labels]))
                neg_feature_label_corr = self.soft_HGR_correlation(f_tav_embs, neg_label_embs)
                feature_label_corrs_list.append(neg_feature_label_corr)
        else:
            pos_feature_label_corr = self.soft_HGR_correlation(f_t_embs, batch_label_embs)
            feature_label_corrs_list.append(pos_feature_label_corr)
            for neg_labels in shuffled_all_neg_labels:
                neg_label_embs = torch.stack(([label_embs[neg_label] for neg_label in neg_labels]))
                neg_feature_label_corr = self.soft_HGR_correlation(f_t_embs, neg_label_embs)
                feature_label_corrs_list.append(neg_feature_label_corr)
        
        feature_label_corrs = torch.stack(([corr for corr in feature_label_corrs_list]))
        feature_label_probs = self.corr_to_prob(feature_label_corrs)

        batch_sample_weights = self.sample_weight()

        if self.pos_aug_param == True:
            positive_pairs_loss_weight = torch.FloatTensor([1/3, 2/3, 2/3, 1.0]).reshape(-1, 1).expand_as(torch.zeros(4, feature_label_probs[:4].shape[-1])).to(self.device).clone()
            feature_label_pos_loss = torch.mean(self.positive_pairs_loss(feature_label_probs[:4]) * positive_pairs_loss_weight, dim = 0)
            feature_label_neg_loss = torch.mean(self.negative_pairs_loss(feature_label_probs[4:]), dim = 0)
            feature_label_loss = - torch.mean(batch_sample_weights * (feature_label_pos_loss + feature_label_neg_loss))
        else:
            feature_label_pos_loss = self.positive_pairs_loss(feature_label_probs[0])
            feature_label_neg_loss = torch.mean(self.negative_pairs_loss(feature_label_probs[1:]), dim = 0)
            feature_label_loss = - torch.mean(batch_sample_weights * (feature_label_pos_loss + feature_label_neg_loss))
        

        # label-label discrimination loss
        label_label_corrs_list = []
        pos_label_label_corr = torch.zeros(self.num_classes).to(self.device)
        label_label_corrs_list.append(pos_label_label_corr)

        extended_labels = self.label_indices.expand_as(torch.zeros(self.num_classes, self.num_classes)).to(self.device).transpose(1, 0).clone()
        extended_labels[0] = self.label_indices
        for i in range(self.num_classes):
            extended_labels[i][i] = 0
        extended_neg_labels = extended_labels[1:]

        for neg_labels in extended_neg_labels:
            neg_label_embs = torch.stack(([label_embs[label] for label in neg_labels]))
            neg_label_label_corr = self.dot_product_similarity(label_embs, neg_label_embs)
            label_label_corrs_list.append(neg_label_label_corr)
        
        label_label_corrs = torch.stack(([corr for corr in label_label_corrs_list]))
        label_label_probs = self.corr_to_prob(label_label_corrs)

        label_label_neg_loss = torch.mean(self.negative_pairs_loss(label_label_probs[1:]), dim = 0)
        label_label_loss = - torch.mean(label_label_neg_loss)
        
        loss = feature_label_loss + self.label_loss_param * label_label_loss

        return loss
        



'''
The label embedding network is designed to learn dense embeddings for discrete emotion categories, which
consists of an embedding layer and a fully-connected layer.
'''
class LabelEmbedding(nn.Module):

    def __init__(self, num_classes, hidden_dim, output_dim):
        super().__init__()

        self.embedding_layer = nn.Embedding(num_classes, hidden_dim)
        self.linear_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    

    def forward(self, labels):
        return self.dropout(self.linear_layer(self.relu(self.embedding_layer(labels))))