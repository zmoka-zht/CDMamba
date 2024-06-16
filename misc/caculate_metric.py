import torch
import numpy as np

#segment metric

def calcuate_confusion_matrix(num_class:int, gt:torch.tensor, pred:torch.tensor):
    '''

    :param num_class:
    :param gt:
    :param pred:
    :return:
    '''
    gt_vector = gt.flatten()
    pred_vector = pred.flatten()
    #验证标签值是否正确
    mask = (gt_vector >= 0) & (gt_vector < num_class)
    #计算混淆矩阵
    cm = torch.bincount(num_class * gt_vector[mask].to(dtype=int) + pred_vector[mask], minlength=num_class ** 2).reshape(num_class, num_class)
    return cm

class segmengtion_metric(object):
    def __init__(self, num_class:int, device:str):
        self.num_class = num_class
        self.device = device
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class)).to(self.device)
        #self.confusion_matrix = torch.zeros((self.num_class, self.num_class))

    def clear(self):
        self.confusion_matrix = torch.zeros((self.num_class, self.num_class)).to(self.device)
        #self.confusion_matrix = torch.zeros((self.num_class, self.num_class))

    def update_confusion_matrix(self, gt, pred):
        cm = calcuate_confusion_matrix(self.num_class, gt, pred)
        self.confusion_matrix += cm

    def get_matrix_per_batch(self, gt, pred):
        confusion_matrix = calcuate_confusion_matrix(self.num_class, gt, pred)
        # 真正例 tp
        tp = torch.diag(confusion_matrix)
        # 全部真实标签 tp+fn
        sum_a1 = torch.sum(confusion_matrix, dim=1)
        #全部预测为正例标签 tp+fp
        sum_a0 = torch.sum(confusion_matrix, dim=0)

        # accuracy
        acc = tp.sum() / (confusion_matrix.sum() + torch.finfo(type=torch.float32).eps)
        # recall
        recall = tp / (sum_a1 + torch.finfo(type=torch.float32).eps)
        # precision
        precision = tp / (sum_a0 + torch.finfo(type=torch.float32).eps)
        # F1 score
        f1 = (2 * recall * precision) / (recall + precision + torch.finfo(type=torch.float32).eps)
        # IoU
        iou = tp / (sum_a1 + sum_a0 - tp + torch.finfo(type=torch.float32).eps)

        #each class metric
        cls_precision = dict(zip(['pre_class[{}]'.format(i) for i in range(self.num_class)], precision))
        cls_recall = dict(zip(['rec_class[{}]'.format(i) for i in range(self.num_class)], recall))
        cls_f1 = dict(zip(['f1_class[{}]'.format(i) for i in range(self.num_class)], f1))
        cls_iou = dict(zip(['iou_class[{}]'.format(i) for i in range(self.num_class)], iou))

        #average metric, ingore non-exist classes
        mean_precision = precision[precision != 0].mean()
        mean_recall = recall[recall != 0].mean()
        mean_iou = iou[iou != 0].mean()
        mean_f1 = f1[f1 != 0].mean()

        #save metric dict
        score_dict_batch = {'acc': acc, 'mean_pre': mean_precision, 'mean_rec': mean_recall, 'mIoU': mean_iou, 'mF1': mean_f1}
        score_dict_batch.update(cls_precision)
        score_dict_batch.update(cls_recall)
        score_dict_batch.update(cls_iou)
        score_dict_batch.update(cls_f1)

        return score_dict_batch

    def get_metric_dict_per_epoch(self):
        # true positive
        tp = torch.diag(self.confusion_matrix)
        # the whole groundtruth registered
        sum_a1 = torch.sum(self.confusion_matrix, dim=1)
        # the whole "truth" predicted
        sum_a0 = torch.sum(self.confusion_matrix, dim=0)
        # accuracy
        acc = tp.sum() / (self.confusion_matrix.sum() + torch.finfo(type=torch.float32).eps)
        # recall
        recall = tp / (sum_a1 + torch.finfo(type=torch.float32).eps)
        # precision
        precision = tp / (sum_a0 + torch.finfo(type=torch.float32).eps)
        # F1 score
        f1 = (2 * recall * precision) / (recall + precision + torch.finfo(type=torch.float32).eps)
        # IoU
        iou = tp / (sum_a1 + sum_a0 - tp + torch.finfo(type=torch.float32).eps)

        # metrics for each class
        cls_precision = dict(zip(['Precision_Class[{}]'.format(i) for i in range(self.num_class)], precision))
        cls_recall = dict(zip(['Recall_Class[{}]'.format(i) for i in range(self.num_class)], recall))
        cls_iou = dict(zip(['IoU_Class[{}]'.format(i) for i in range(self.num_class)], iou))
        cls_f1 = dict(zip(['F1_Class[{}]'.format(i) for i in range(self.num_class)], f1))

        # average metrics
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        mean_iou = iou.mean()
        mean_f1 = f1.mean()
        score_dict_epoch = {'Accuracy': acc, 'mean_Precision': mean_precision, 'mean_Recall': mean_recall,
                            'mIoU': mean_iou, 'mF1': mean_f1}

        # save metric dict
        score_dict_epoch.update(cls_precision)
        score_dict_epoch.update(cls_recall)
        score_dict_epoch.update(cls_iou)
        score_dict_epoch.update(cls_f1)
        return score_dict_epoch






if __name__=="__main__":
    gt_label = torch.tensor([[0, 1, 2, 3, 1],
                         [1, 2, 2, 3, 4]])

    pre_label = torch.tensor([[0, 1, 2, 3, 1],
                          [5, 1, 2, 1, 4]])

    num_class = 6
    metric = segmengtion_metric(6, 'cuda:0')
    res = metric.get_matrix_per_batch(gt_label, pre_label)
    res1 = metric.get_metric_dict_per_epoch()
    print(res)
    print(res1)
