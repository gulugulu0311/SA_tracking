import numpy as np

eps = np.finfo(np.float32).eps.item()


class Evaluator(object):
    """
    Standard Pixel-level Semantic Segmentation Evaluator.
    Calculates OA, AA, mIoU, F1, Kappa based on the Confusion Matrix.
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):  # Overall Accuracy
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Class_Accuracy(self):
        # Recall per class (Producer's Accuracy for pixels)
        Acc_classes = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + eps)
        Acc = np.nanmean(Acc_classes)
        return Acc_classes, Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + eps)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def F1(self):
        precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + eps)
        recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1 = np.nanmean(f1)
        return f1

    def Kappa(self):
        p_o = self.Pixel_Accuracy()
        pre = np.sum(self.confusion_matrix, axis=0)
        label = np.sum(self.confusion_matrix, axis=1)
        p_e = (pre * label).sum() / (self.confusion_matrix.sum() * self.confusion_matrix.sum() + eps)
        kappa = (p_o - p_e) / (1 - p_e + eps)
        return kappa

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
    """
    Evaluates Binary Change Detection (Change vs No-Change) in spatial domain.
    Also tracks a simple Land Cover Change (LCC) exact match accuracy.
    """
    def __init__(self):
        self.spatial_f1 = None
        self.spatial_ua = None
        self.spatial_pa = None
        # Counters for binary confusion matrix
        self.PreChange_LabChange = eps      # TP
        self.PreNoChange_LabChange = eps    # FN
        self.PreChange_LabNoChange = eps    # FP
        self.PreNoChange_LabNoChange = eps  # TN
        # LCC accuracy counters
        self.lcc_nume = eps
        self.lcc_deno = eps
        self.tolerance = 1

    def addValue(self, label, pre):
        # If list is not empty, it implies change points exist
        has_change_label = len(label) != 0
        has_change_pre = len(pre) != 0
        
        if has_change_label and has_change_pre:
            self.PreChange_LabChange += 1
        elif not has_change_label and not has_change_pre:
            self.PreNoChange_LabNoChange += 1
        elif not has_change_label and has_change_pre:
            self.PreChange_LabNoChange += 1
        elif has_change_label and not has_change_pre:
            self.PreNoChange_LabChange += 1
         
    def getScore(self):
        # Calculate UA (Precision) and PA (Recall) for 'Change' class
        self.spatial_ua_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreChange_LabNoChange)
        self.spatial_ua_Nochange = self.PreNoChange_LabNoChange / (self.PreNoChange_LabNoChange + self.PreNoChange_LabChange)

        self.spatial_pa_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreNoChange_LabChange)
        self.spatial_pa_Nochange = self.PreNoChange_LabNoChange / (self.PreNoChange_LabNoChange + self.PreChange_LabNoChange)

        self.spatial_pa = (self.spatial_pa_change + self.spatial_pa_Nochange) / 2
        self.spatial_ua = (self.spatial_ua_change + self.spatial_ua_Nochange) / 2
        self.spatial_f1 = 2 * self.spatial_pa * self.spatial_ua / (self.spatial_pa + self.spatial_ua + eps)
        
    # def addLccValue(self, pretypes, labeltypes):
    #     # S-SMA Logic: Exact match of the entire sequence of types
    #     if np.array_equal(pretypes, labeltypes):
    #         self.lcc_nume += 1
    #     self.lcc_deno += 1
    def addLccValue(self, pretypes, labeltypes, prepoints=None, labelpoints=None):
        if not np.array_equal(pretypes, labeltypes):
            self.lcc_deno += 1
            return

        if prepoints is not None and labelpoints is not None:
            if len(prepoints) != len(labelpoints):
                self.lcc_deno += 1
                return
            if len(prepoints) == 0:
                self.lcc_nume += 1
                self.lcc_deno += 1
                return

            time_diff = np.abs(prepoints - labelpoints)
            
            if np.all(time_diff <= self.tolerance):
                self.lcc_nume += 1
            
            self.lcc_deno += 1
            
        else:
            self.lcc_nume += 1
            self.lcc_deno += 1
    
    def getLccScore(self):
        return self.lcc_nume / self.lcc_deno


class TemporalChangeDetectScore(object):
    """
    Evaluates Change Timing Accuracy with a tolerance window (error_rate).
    """
    def __init__(self, series_length=60, error_rate=0):
        self.temporal_f1 = None
        self.temporal_ua = None
        self.temporal_pa = None
        
        self.PreChange_LabChange = eps
        self.PreNoChange_LabChange = eps
        self.PreChange_LabNoChange = eps
        self.PreNoChange_LabNoChange = eps
        
        self.series_length = series_length
        self.error_rate = error_rate
        
        # Exact CD match accuracy (T-SMA counters)
        self.cd_nume = eps
        self.cd_deno = eps

    def addValue(self, label, pre):
        # Create a copy to modify for matching
        pre_matched = list(pre)
        
        # Relaxed matching: Snap prediction to label if within tolerance
        for lab in label:
            for p_index in range(len(pre_matched)):
                if abs(pre_matched[p_index] - lab) <= self.error_rate:
                    pre_matched[p_index] = lab
        
        better_pre = sorted(list(set(pre_matched))) # Remove duplicates after snapping and sort
        
        # Check for exact match after relaxation (T-SMA logic)
        if np.array_equal(better_pre, label):
            self.cd_nume += 1
        self.cd_deno += 1
        
        # Calculate Temporal Binary Metrics (Time-step level)
        hot_label = np.zeros(self.series_length)
        if len(label) != 0:
            hot_label[np.array(label)] = 1 
            
        hot_pre = np.zeros(self.series_length)
        if len(better_pre) != 0:
            hot_pre[np.array(better_pre)] = 1
            
        self.hot_label = hot_label
        self.hot_pre = hot_pre
        
        self.PreChange_LabChange += np.sum((hot_pre == 1) & (hot_label == 1))
        self.PreNoChange_LabChange += np.sum((hot_pre != 1) & (hot_label == 1))
        self.PreChange_LabNoChange += np.sum((hot_pre == 1) & (hot_label != 1))
        self.PreNoChange_LabNoChange += np.sum((hot_pre != 1) & (hot_label != 1))

    def getScore(self):
        self.temporal_ua_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreChange_LabNoChange)
        self.temporal_ua_Nochange = self.PreNoChange_LabNoChange / (self.PreNoChange_LabNoChange + self.PreNoChange_LabChange)

        self.temporal_pa_change = self.PreChange_LabChange / (self.PreChange_LabChange + self.PreNoChange_LabChange)
        self.temporal_pa_Nochange = self.PreNoChange_LabNoChange / (self.PreNoChange_LabNoChange + self.PreChange_LabNoChange)

        self.temporal_pa = (self.temporal_pa_change + self.temporal_pa_Nochange) / 2
        self.temporal_ua = (self.temporal_ua_change + self.temporal_ua_Nochange) / 2

        self.temporal_f1 = 2 * self.temporal_pa * self.temporal_ua / (self.temporal_pa + self.temporal_ua + eps)
    
    def getCDScore(self):
        return self.cd_nume / self.cd_deno


class ChangeTypeAccuracyMatrix:
    """
    Enhanced Change Type Accuracy Evaluator.
    Calculates PA (Recall), UA (Precision), and F1-score for each transition type (i -> j).
    
    Logic:
        - Extracts 'events' as (time, from_class, to_class).
        - Matches predicted events to ground truth events with temporal tolerance.
        - Tracks True Positives (TP), False Negatives (FN -> via gt_counts), and False Positives (FP -> via pred_counts).
    
    Args:
        num_classes: Number of land cover classes.
        tol: Temporal tolerance for event matching (e.g., +/- 1 month).
    """
    def __init__(self, num_classes, tol=1):
        self.num_classes = num_classes
        self.tol = tol
        
        # TP: Successfully matched events
        self.numer = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # GT Total: Total real events (TP + FN) -> Denominator for Producer's Accuracy (Recall)
        self.gt_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # Pred Total: Total predicted events (TP + FP) -> Denominator for User's Accuracy (Precision)
        self.pred_counts = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.numer.fill(0)
        self.gt_counts.fill(0)
        self.pred_counts.fill(0)

    def _extract_events(self, seq):
        """
        Extract change events from a label sequence.
        Returns: list of tuples (time_index, from_class, to_class)
        """
        seq = np.asarray(seq)
        events = []
        # Iterate from index 1 to end
        for t in range(1, len(seq)):
            if seq[t] != seq[t-1]:
                events.append((t, int(seq[t-1]), int(seq[t])))
        return events

    def add_sequence(self, gt_seq, pred_seq):
        """
        Process a single sample's GT and Prediction sequences.
        """
        true_events = self._extract_events(gt_seq)
        pred_events = self._extract_events(pred_seq)

        # 1. Update Ground Truth Counters (Denominator for PA)
        for _, f, t in true_events:
            self.gt_counts[f, t] += 1

        # 2. Update Prediction Counters (Denominator for UA)
        for _, f, t in pred_events:
            self.pred_counts[f, t] += 1

        # 3. Match Logic (To find True Positives)
        if len(pred_events) == 0:
            return # No predictions, no TP possible

        pred_times = np.array([e[0] for e in pred_events])
        pred_froms = np.array([e[1] for e in pred_events])
        pred_tos = np.array([e[2] for e in pred_events])
        
        used_pred_indices = set() # Keep track of matched predictions to avoid double counting

        for t_true, f_true, to_true in true_events:
            # Find candidate predictions within time tolerance
            diffs = np.abs(pred_times - t_true)
            cand_indices = np.where(diffs <= self.tol)[0]

            if cand_indices.size == 0:
                continue

            # Sort candidates: prioritize closest time, then smallest index (stable sort)
            # This logic mimics finding the "best match" for the current GT event
            cand_sorted = sorted(cand_indices, key=lambda k: (abs(pred_times[k] - t_true), k))
            
            chosen_idx = None
            for idx in cand_sorted:
                if idx not in used_pred_indices:
                    chosen_idx = idx
                    break
            
            if chosen_idx is None:
                continue # All candidates already used

            # Mark this prediction as used
            used_pred_indices.add(chosen_idx)

            # Check if the transition type matches (from -> to)
            if pred_froms[chosen_idx] == f_true and pred_tos[chosen_idx] == to_true:
                self.numer[f_true, to_true] += 1

    def get_metrics_matrices(self):
        """
        Calculate and return PA, UA, and F1 matrices.
        Returns dictionary containing the matrices.
        """
        # Producer's Accuracy (Recall) = TP / GT_Total
        pa_matrix = np.zeros_like(self.numer, dtype=float)
        mask_gt = (self.gt_counts > 0)
        pa_matrix[mask_gt] = self.numer[mask_gt] / self.gt_counts[mask_gt]

        # User's Accuracy (Precision) = TP / Pred_Total
        ua_matrix = np.zeros_like(self.numer, dtype=float)
        mask_pred = (self.pred_counts > 0)
        ua_matrix[mask_pred] = self.numer[mask_pred] / self.pred_counts[mask_pred]

        # F1 Score = 2 * PA * UA / (PA + UA)
        f1_matrix = np.zeros_like(self.numer, dtype=float)
        sum_paua = pa_matrix + ua_matrix
        mask_f1 = (sum_paua > 0)
        f1_matrix[mask_f1] = 2 * pa_matrix[mask_f1] * ua_matrix[mask_f1] / sum_paua[mask_f1]
        
        total_gt_samples = np.sum(self.gt_counts)
        if total_gt_samples > 0:
            
            weighted_sum = np.sum(f1_matrix * self.gt_counts)
            weighted_f1 = weighted_sum / total_gt_samples
        else:
            weighted_f1 = 0.0
        
        return {
            "PA": pa_matrix,
            "UA": ua_matrix,
            "F1": f1_matrix,
            "TP_Counts": self.numer,
            "GT_Counts": self.gt_counts,
            "Pred_Counts": self.pred_counts,
            "Weighted_F1": weighted_f1
        }