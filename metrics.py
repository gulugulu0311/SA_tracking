"""
@Author ：hhx
@Description ：classification and change detection metrics
"""

import numpy as np

eps = np.finfo(np.float32).eps.item()


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):  # Overall Accuracy
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Class_Accuracy(self):
        Acc_classes = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc_classes)
        return Acc_classes, Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def F1(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
        f1 = 2 * precision * recall / (precision + recall)
        f1 = np.nanmean(f1)
        return f1

    def Kappa(self):
        p_o = self.Pixel_Accuracy()
        pre = np.sum(self.confusion_matrix, axis=0)
        label = np.sum(self.confusion_matrix, axis=1)
        p_e = (pre * label).sum() / (self.confusion_matrix.sum() * self.confusion_matrix.sum())
        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class SpatialChangeDetectScore(object):
    def __init__(self):
        self.spatial_f1 = None
        self.spatial_ua_Nochange = None
        self.spatial_ua_change = None
        self.spatial_pa_Nochange = None
        self.spatial_pa_change = None
        self.PreChange_LabChange = eps
        self.PreNoChange_LabChange = eps
        self.PreChange_LabNoChange = eps
        self.PreNoChange_LabNoChange = eps
        # lcc accuracy
        self.lcc_nume = eps
        self.lcc_deno = eps

    def addValue(self, label, pre):
        if len(label) != 0 and len(pre) != 0:
            self.PreChange_LabChange += 1
        elif len(label) == 0 and len(pre) == 0:
            self.PreNoChange_LabNoChange += 1
        elif len(label) == 0 and len(pre) != 0:
            self.PreChange_LabNoChange += 1
        elif len(label) != 0 and len(pre) == 0:
            self.PreNoChange_LabChange += 1
         
    def getScore(self):
        self.spatial_ua_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreChange_LabNoChange)
        self.spatial_ua_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreNoChange_LabChange)

        self.spatial_pa_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreNoChange_LabChange)
        self.spatial_pa_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreChange_LabNoChange)

        self.spatial_pa = (self.spatial_pa_change + self.spatial_pa_Nochange) / 2
        self.spatial_ua = (self.spatial_ua_change + self.spatial_ua_Nochange) / 2
        self.spatial_f1 = 2 * self.spatial_pa * self.spatial_ua / (
                self.spatial_pa + self.spatial_ua)
        
    def addLccValue(self, pretypes, labeltypes):
        if np.array_equal(pretypes, labeltypes):
            self.lcc_nume += 1
            self.lcc_deno += 1
        else:
            self.lcc_deno += 1
    
    def getLccScore(self):
        return self.lcc_nume / self.lcc_deno

class TemporalChangeDetectScore(object):
    def __init__(self, series_length=60, error_rate=0):
        self.temporal_f1 = None
        self.temporal_ua_Nochange = None
        self.temporal_ua_change = None
        self.temporal_pa_Nochange = None
        self.temporal_pa_change = None

        self.PreChange_LabChange = eps
        self.PreNoChange_LabChange = eps
        self.PreChange_LabNoChange = eps
        self.PreNoChange_LabNoChange = eps
        self.series_length = series_length
        self.error_rate = error_rate
        
        # CD accuracy
        self.cd_nume = eps
        self.cd_deno = eps

    def addValue(self, label, pre):
        for lab in label:
            for p_index in range(len(pre)):
                if abs(pre[p_index] - lab) <= self.error_rate:
                    pre[p_index] = lab
        better_pre = list(set(pre))  # 去重
        if np.array_equal(better_pre, label):
            self.cd_nume += 1
            self.cd_deno += 1
        else:
            self.cd_deno += 1
        hot_label = np.zeros(self.series_length)
        if len(label) != 0:
            hot_label[np.array(label)] = 1  # 标签
        hot_pre = np.zeros(self.series_length)
        if len(better_pre) != 0:
            hot_pre[np.array(better_pre)] = 1  # 预测
            
        self.hot_label = hot_label
        self.hot_pre = hot_pre
        self.PreChange_LabChange += np.where((hot_pre == 1) & (hot_label == 1))[0].shape[0]
        self.PreNoChange_LabChange += np.where((hot_pre != 1) & (hot_label == 1))[0].shape[0]
        self.PreChange_LabNoChange += np.where((hot_pre == 1) & (hot_label != 1))[0].shape[0]
        self.PreNoChange_LabNoChange += np.where((hot_pre != 1) & (hot_label != 1))[0].shape[0]

    def getScore(self):
        self.temporal_ua_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreChange_LabNoChange)
        self.temporal_ua_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreNoChange_LabChange)

        self.temporal_pa_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreNoChange_LabChange)
        self.temporal_pa_Nochange = self.PreNoChange_LabNoChange / (
                self.PreNoChange_LabNoChange + self.PreChange_LabNoChange)

        self.temporal_pa = (self.temporal_pa_change + self.temporal_pa_Nochange) / 2
        self.temporal_ua = (self.temporal_ua_change + self.temporal_ua_Nochange) / 2

        self.temporal_f1 = 2 * self.temporal_pa * self.temporal_ua / (
                self.temporal_pa + self.temporal_ua)
    
    def getCDScore(self):
        return self.cd_nume / self.cd_deno

import numpy as np

class ChangeTypeAccuracyMatrix:
    """
    统计每一种变化类型 i→j 的准确率：
    对每一个真实变化事件 i→j，分母 denom[i,j] += 1
    若预测事件也是 i→j，则分子 numer[i,j] += 1

    参数：
        num_classes: 类别数量
        tol: 时间容差（论文 = ±2）
    """
    def __init__(self, num_classes, tol=1):
        self.num_classes = num_classes
        self.tol = tol
        self.numer = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.denom = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.numer.fill(0)
        self.denom.fill(0)

    def _extract_events(self, seq):
        """
        从序列中提取所有变化事件，格式为：
        (t, from_class, to_class)
        """
        seq = np.asarray(seq)
        events = []
        for t in range(1, len(seq)):
            if seq[t] != seq[t-1]:
                events.append((t, int(seq[t-1]), int(seq[t])))
        return events

    def add_sequence(self, gt_seq, pred_seq):
        """
        对一条序列进行变化类型准确率统计。
        gt_seq / pred_seq: 一维类别数组
        """

        true_events = self._extract_events(gt_seq)
        pred_events = self._extract_events(pred_seq)

        if len(pred_events) > 0:
            pred_times = np.array([e[0] for e in pred_events])
            pred_froms = np.array([e[1] for e in pred_events])
            pred_tos = np.array([e[2] for e in pred_events])
        else:
            pred_times = np.array([], dtype=int)

        used_pred = set()

        for t_true, f_true, to_true in true_events:

            # 所有真实事件 i→j 的 denom 加 1
            self.denom[f_true, to_true] += 1

            # 时间容差范围内找预测事件
            if len(pred_times) == 0:
                continue

            diffs = np.abs(pred_times - t_true)
            cand = np.where(diffs <= self.tol)[0]

            if cand.size == 0:
                continue

            # 找最接近的未使用预测事件
            cand_sorted = sorted(cand, key=lambda k: (abs(pred_times[k]-t_true), k))
            chosen = None
            for c in cand_sorted:
                if c not in used_pred:
                    chosen = c
                    break
            
            if chosen is None:
                continue

            used_pred.add(chosen)

            # 若预测变化类型也等于真实变化类型，则 numer +1
            if pred_froms[chosen] == f_true and pred_tos[chosen] == to_true:
                self.numer[f_true, to_true] += 1

    def get_accuracy_matrix(self):
        """ 返回逐类型准确率的矩阵 """
        accuracy = np.zeros_like(self.numer, dtype=float)
        mask = (self.denom > 0)
        accuracy[mask] = self.numer[mask] / self.denom[mask]
        return accuracy