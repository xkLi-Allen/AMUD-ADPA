import math
import torch
import random
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from sklearn.cluster import KMeans
from tasks.clustering_metrics import clustering_metrics
from sklearn.metrics import roc_auc_score, average_precision_score



def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return (correct / len(labels)).item()

def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def add_labels(features, labels, idx, num_classes):
    onehot = np.zeros([features.shape[0], num_classes])
    onehot[idx, labels[idx]] = 1
    return np.concatenate([features, onehot], axis=-1)

def link_cls_train(model, train_query_edges, train_labels, device, optimizer, loss_fn):
    model.train()
    model.base_model.query_edges = train_query_edges
    optimizer.zero_grad()
    train_output = model.model_forward(None, device)
    loss_train = loss_fn(train_output, train_labels)
    acc_train = accuracy(train_output, train_labels)
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train

def link_cls_mini_batch_train(model, train_query_edges, train_loader, train_labels, device, optimizer, loss_fn):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        train_idx = torch.unique(train_query_edges[batch].reshape(-1))
        node_idx_map_dict = {train_idx[i].item():i for i in range(len(torch.unique(train_query_edges[batch].reshape(-1))))}
        row,col = train_query_edges[batch].T
        row = torch.tensor([node_idx_map_dict[row[i].item()] for i in range(len(row))])
        col = torch.tensor([node_idx_map_dict[col[i].item()] for i in range(len(col))])
        model.base_model.query_edges = torch.stack((row,col)).T
        train_output = model.model_forward(train_idx, device, train_query_edges[batch])
        loss_train = loss_fn(train_output, train_labels[batch])
        pred = train_output.max(1)[1].type_as(train_labels)
        correct_num += pred.eq(train_labels[batch]).double().sum()
        loss_train_sum += loss_train.item()
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_query_edges)

    return loss_train, acc_train.item()
    
def link_cls_evaluate(model, val_query_edges, test_query_edges, val_labels, test_labels, device):
    model.eval()
    model.base_model.query_edges = val_query_edges
    val_output = model.model_forward(None, device)
    model.base_model.query_edges = test_query_edges
    test_output = model.model_forward(None, device)
    acc_val = accuracy(val_output, val_labels)
    acc_test = accuracy(test_output, test_labels)
    return acc_val, acc_test

def link_cls_mini_batch_evaluate(model, val_query_edges, val_loader, test_query_edges, test_loader, val_labels, test_labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_idx = torch.unique(val_query_edges[batch].reshape(-1))
        node_idx_map_dict = {val_idx[i].item():i for i in range(len(torch.unique(val_query_edges[batch].reshape(-1))))}
        row,col = val_query_edges[batch].T
        row = torch.tensor([node_idx_map_dict[row[i].item()] for i in range(len(row))])
        col = torch.tensor([node_idx_map_dict[col[i].item()] for i in range(len(col))])
        model.base_model.query_edges = torch.stack((row,col)).T
        val_output = model.model_forward(val_idx, device, val_query_edges[batch])
        pred = val_output.max(1)[1].type_as(val_labels)
        correct_num_val += pred.eq(val_labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_query_edges)

    for batch in test_loader:
        test_idx = torch.unique(test_query_edges[batch].reshape(-1))
        node_idx_map_dict = {test_idx[i].item():i for i in range(len(torch.unique(test_query_edges[batch].reshape(-1))))}
        row,col = test_query_edges[batch].T
        row = torch.tensor([node_idx_map_dict[row[i].item()] for i in range(len(row))])
        col = torch.tensor([node_idx_map_dict[col[i].item()] for i in range(len(col))])
        model.base_model.query_edges = torch.stack((row,col)).T
        test_output = model.model_forward(test_idx, device, test_query_edges[batch])
        pred = test_output.max(1)[1].type_as(test_labels)
        correct_num_test += pred.eq(test_labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_query_edges)

    return acc_val.item(), acc_test.item()

def node_cls_evaluate(model, val_idx, test_idx, labels, device):
    model.eval()
    val_output = model.model_forward(val_idx, device)
    test_output = model.model_forward(test_idx, device)
    acc_val = accuracy(val_output, labels[val_idx])
    acc_test = accuracy(test_output, labels[test_idx])
    return acc_val, acc_test


def node_cls_mini_batch_evaluate(model, val_idx, val_loader, test_idx, test_loader, labels, device):
    model.eval()
    correct_num_val, correct_num_test = 0, 0
    for batch in val_loader:
        val_output = model.model_forward(batch, device)
        pred = val_output.max(1)[1].type_as(labels)
        correct_num_val += pred.eq(labels[batch]).double().sum()
    acc_val = correct_num_val / len(val_idx)

    for batch in test_loader:
        test_output = model.model_forward(batch, device)
        pred = test_output.max(1)[1].type_as(labels)
        correct_num_test += pred.eq(labels[batch]).double().sum()
    acc_test = correct_num_test / len(test_idx)

    return acc_val.item(), acc_test.item()


def node_cls_train(model, train_idx, labels, device, optimizer, loss_fn, retain_graph = False):
    model.train()
    optimizer.zero_grad()
    train_output = model.model_forward(train_idx, device)
    #print(train_idx)
    #print(labels[train_idx])
    loss_train = loss_fn(train_output, labels[train_idx])
    acc_train = accuracy(train_output, labels[train_idx])
    loss_train.backward(retain_graph = retain_graph)
    optimizer.step()

    return loss_train.item(), acc_train


def node_cls_mini_batch_train(model, train_idx, train_loader, labels, device, optimizer, loss_fn, retain_graph = False):
    model.train()
    correct_num = 0
    loss_train_sum = 0.
    for batch in train_loader:
        train_output = model.model_forward(batch, device)
        loss_train = loss_fn(train_output, labels[batch])
        pred = train_output.max(1)[1].type_as(labels)
        correct_num += pred.eq(labels[batch]).double().sum()
        loss_train_sum += loss_train.item()
        optimizer.zero_grad()
        loss_train.backward(retain_graph = retain_graph)
        optimizer.step()

    loss_train = loss_train_sum / len(train_loader)
    acc_train = correct_num / len(train_idx)

    return loss_train, acc_train.item()


# def cluster_loss(train_output, y_pred, cluster_centers):

#     for i in range(len(cluster_centers)):
#         if i == 0:
#             dist = torch.norm(train_output - cluster_centers[i], p=2, dim=1, keepdim=True)
#         else:
#             dist = torch.cat((dist, torch.norm(train_output - cluster_centers[i], p=2, dim=1, keepdim=True)), 1)
    
#     loss = 0.
#     loss_tmp = -dist.mean(1).sum()
#     loss_tmp += 2 * np.sum(dist[j, x] for j, x in zip(range(dist.shape[0]), y_pred))
#     loss = loss_tmp / dist.shape[0]
#     return loss


# def clustering_train(model, train_idx, labels, device, optimizer, loss_fn, n_clusters, n_init):
#     model.train()
#     optimizer.zero_grad()

#     train_output = model.model_forward(train_idx, device)
    
#     # calc loss
#     kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
#     y_pred = kmeans.fit_predict(train_output.data.cpu().numpy()) # cluster_label
#     cluster_centers = torch.FloatTensor(kmeans.cluster_centers_).to(device)

#     loss_train = loss_fn(train_output, y_pred, cluster_centers)
#     loss_train.backward()
#     optimizer.step()

#     # calc acc, nmi, adj
#     labels = labels.cpu().numpy()
#     cm = clustering_metrics(labels, y_pred)
#     acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()

#     return loss_train.item(), acc, nmi, adjscore


# def sparse_to_tuple(sparse_mx):
#     if not sp.isspmatrix_coo(sparse_mx):
#         sparse_mx = sparse_mx.tocoo()
#     coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
#     values = sparse_mx.data
#     shape = sparse_mx.shape
#     return coords, values, shape



# # input full edge_features, pos_edges and neg_edges to calc roc_auc, avg_prec score
# def edge_predict_score(edge_feature, pos_edges, neg_edges, threshold):
#     labels = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))))
#     all_edges = torch.cat((pos_edges, neg_edges))
#     edge_pred = edge_feature[all_edges[:, 0], all_edges[:, 1]].reshape(-1)
#     edge_pred = torch.sigmoid(edge_pred)
#     # edge_pred = edge_pred > threshold
#     roc_auc = roc_auc_score(labels, edge_pred)
#     avg_prec = average_precision_score(labels, edge_pred)
#     return roc_auc, avg_prec


# def edge_predict_train(model, train_node_index, with_params, pos_edges, neg_edges, 
#                        device, optimizer, loss_fn, threshold):
#     if with_params is True:
#         model.train()
#         optimizer.zero_grad()

#     train_output = model.model_forward(train_node_index, device)
#     edge_feature = torch.mm(train_output, train_output.t())
#     labels = torch.cat((torch.ones(len(pos_edges)), torch.zeros(len(neg_edges)))).to(device)
#     train_edge = torch.cat((pos_edges, neg_edges)).to(device)
#     edge_pred = edge_feature[train_edge[:, 0], train_edge[:, 1]].reshape(-1)
#     edge_pred = torch.sigmoid(edge_pred)

#     # logger.info("-----------------------------")
#     # logger.info("edge_features:  ", edge_feature[:200])
#     # logger.info("edge_pred:\n", edge_pred[len(pos_edges)-50:len(pos_edges)+50])
#     # logger.info("labels:\n",labels[len(pos_edges)-50:len(pos_edges)+50])
#     # logger.info("-----------------------------")

#     loss = loss_fn(edge_pred, labels)
#     if with_params is True:
#         loss.backward()
#         optimizer.step()

#     labels = labels.cpu().data
#     edge_pred = edge_pred.cpu().data
#     edge_pred = edge_pred > threshold
#     roc_auc = roc_auc_score(labels, edge_pred)
#     avg_prec = average_precision_score(labels, edge_pred)
#     return loss.item(), roc_auc, avg_prec


# def edge_predict_eval(model, train_node_index, val_pos_edges, val_neg_edges, 
#                       test_pos_edges, test_neg_edges, device, threshold):
#     model.eval()
#     train_output = model.model_forward(train_node_index, device)
#     edge_feature = torch.mm(train_output, train_output.t()).cpu().data

#     roc_auc_val, avg_prec_val = edge_predict_score(edge_feature, val_pos_edges, val_neg_edges, threshold)
#     roc_auc_test, avg_prec_test = edge_predict_score(edge_feature, test_pos_edges, test_neg_edges, threshold)

#     return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test


# def mini_batch_edge_predict_train(model, train_node_index, with_params, train_loader, 
#                                   device, optimizer, loss_fn, threshold):
#     if with_params is True:
#         model.train()
#         optimizer.zero_grad()
    
#     loss_train = 0.
#     roc_auc_sum = 0.
#     avg_prec_sum = 0.

#     output = model.model_forward(train_node_index, device)
#     output = output.cpu()
#     edge_feature = torch.mm(output, output.t())
#     edge_feature = torch.sigmoid(edge_feature)

#     for batch, label in train_loader:
#         edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
#         # logger.info("-----------------------------")
#         # logger.info("edge_pred:\n", edge_pred.data[:100])
#         # logger.info("labels:\n",label.data[:100])
#         # logger.info("roc_auc_partial: ",roc_auc_score(label.data, edge_pred.data[:100]))
#         # logger.info("-----------------------------")
#         pred_label = edge_pred > threshold
#         roc_auc_sum += roc_auc_score(label.data, pred_label.data)
#         avg_prec_sum += average_precision_score(label.data, pred_label.data)

#         edge_pred = edge_pred.to(device)
#         label = label.to(device)
#         loss_train += loss_fn(edge_pred, label)

#     if with_params is True:
#         loss_train.backward()
#         optimizer.step()
        
#     loss_train = loss_train.item() / len(train_loader)
#     roc_auc = roc_auc_sum / len(train_loader)
#     avg_prec = avg_prec_sum / len(train_loader)

#     return loss_train, roc_auc, avg_prec


# def mini_batch_edge_predict_eval(model, train_node_index, val_loader, test_loader, device, threshold):
#     model.eval()
#     roc_auc_val_sum, avg_prec_val_sum = 0., 0.
#     roc_auc_test_sum, avg_prec_test_sum = 0., 0.

#     output = model.model_forward(train_node_index, device)
#     output = output.cpu().data
#     edge_feature = torch.mm(output, output.t())
#     edge_feature = torch.sigmoid(edge_feature)

#     for batch, label in val_loader:
#         edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
#         label_pred = edge_pred > threshold
#         roc_auc_val_sum += roc_auc_score(label, label_pred)
#         avg_prec_val_sum += average_precision_score(label, label_pred)

#     roc_auc_val = roc_auc_val_sum / len(val_loader)
#     avg_prec_val = avg_prec_val_sum / len(val_loader)

#     for batch, label in test_loader:
#         edge_pred = edge_feature[batch[:, 0], batch[:, 1]].reshape(-1)
#         label_pred = edge_pred > threshold
#         roc_auc_test_sum += roc_auc_score(label, edge_pred)
#         avg_prec_test_sum += average_precision_score(label, edge_pred)

#     roc_auc_test = roc_auc_test_sum / len(test_loader)
#     avg_prec_test = avg_prec_test_sum / len(test_loader)

#     return roc_auc_val, avg_prec_val, roc_auc_test, avg_prec_test


# def mix_pos_neg_edges(pos_edges, neg_edges, mix_size):
#     start, end = 0, mix_size
#     mix_edges = torch.cat((pos_edges[start:end], neg_edges[start:end]))
#     mix_labels = torch.cat((torch.ones(end - start), torch.zeros(end - start)))

#     start += mix_size
#     end += mix_size
#     while end < len(pos_edges):
#         tmp_edges = torch.cat((pos_edges[start:end], neg_edges[start:end]))
#         tmp_labels = torch.cat((torch.ones(end - start), torch.zeros(end - start)))
#         mix_edges = torch.cat((mix_edges, tmp_edges))
#         mix_labels = torch.cat((mix_labels, tmp_labels))
#         start += mix_size
#         end += mix_size
    
#     tmp_edges = torch.cat((pos_edges[start:], neg_edges[start:]))
#     tmp_labels = torch.cat((torch.ones(len(pos_edges) - start), torch.zeros(len(neg_edges) - start)))
#     mix_edges = torch.cat((mix_edges, tmp_edges))
#     mix_labels = torch.cat((mix_labels, tmp_labels))

#     return mix_edges, mix_labels

    
