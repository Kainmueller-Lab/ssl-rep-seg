import numpy as np
from tqdm import tqdm
class PositionalBiasEvaluator:
    def __init__(self, n_classes, n_bins=16):
        self.n_classes = n_classes
        self.n_bins = n_bins

        self.H_gt_reference = None
        self.MI_gt = None
        self.classwise_MI_gt = None

    # --------------------------------------------------
    # Core utilities
    # --------------------------------------------------

    def _build_position_class_counts(self, masks):
        H, W = masks[0].shape
        B = self.n_bins * self.n_bins

        counts = np.zeros((B, self.n_classes), dtype=np.float64)

        xs = (np.arange(W) * self.n_bins / W).astype(int)
        ys = (np.arange(H) * self.n_bins / H).astype(int)

        for m in masks:
            for y in range(H):
                for x in range(W):
                    bin_id = ys[y] * self.n_bins + xs[x]
                    c = int(m[y, x])  
                    if c < 0: # for GT mask, -1 is ignore index, skip it
                        pass
                    else:
                        counts[bin_id, c] += 1

        return counts

    def _mutual_information_from_counts(self, counts):
        total = counts.sum()
        P = counts / total

        Pp = P.sum(axis=1, keepdims=True) #shape (B, 1)
        Pc = P.sum(axis=0, keepdims=True) #shape (1, C)
        PpPc = Pp @ Pc # matrix multiplication, shape (B, C)
        
        classwise_mi = np.zeros(self.n_classes)
        global_mi = 0.0

        for p in range(P.shape[0]):
            for c in range(P.shape[1]):
                if P[p, c] > 0:
                    term = P[p, c] * np.log(P[p, c] / PpPc[p, c])
                    classwise_mi[c] += term
                    global_mi += term

        return global_mi, classwise_mi, Pc.flatten()

    @staticmethod
    def _entropy_from_probs(P):
        P_nonzero = P[P > 0]
        return -np.sum(P_nonzero * np.log(P_nonzero))

    def build_avg_var_cluster_maps(
            self, masks, clusters=None, avg_maps=None, var_maps=None):
        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise ValueError("masks must have shape (N, H, W).")

        if clusters is None:
            clusters = np.arange(self.n_classes)

        H, W = masks.shape[1], masks.shape[2]

        if avg_maps is None:
            avg_maps = np.zeros((self.n_classes, H, W), dtype=np.float64)
        if var_maps is None:
            var_maps = np.zeros((self.n_classes, H, W), dtype=np.float64)

        iterator = clusters
        if "tqdm" in globals():
            iterator = tqdm(clusters, total=len(clusters))

        for c in iterator:
            bin_masks = (masks == c).astype(np.float64)
            avg_maps[c] = bin_masks.mean(axis=0)
            var_maps[c] = bin_masks.var(axis=0)

        return avg_maps, var_maps

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def fit_ground_truth(self, gt_masks=None, gt_counts=None):
        if gt_counts is None:
            gt_counts = self._build_position_class_counts(gt_masks)

        MI_gt, classwise_mi_gt, Pc_gt = \
            self._mutual_information_from_counts(gt_counts)

        H_gt = self._entropy_from_probs(Pc_gt)

        self.H_gt_reference = H_gt
        self.MI_gt = MI_gt
        self.classwise_MI_gt = classwise_mi_gt

        return {
            "GT_MI": MI_gt,
            "GT_NMI": MI_gt/H_gt if H_gt > 0 else 0.0,
            "GT_entropy": H_gt,
            "GT_classwise_MI": classwise_mi_gt,
        }

    def evaluate_predictions(self, pred_masks=None, pred_counts=None):
        if self.H_gt_reference is None:
            raise RuntimeError("Call fit_ground_truth() first.")

        if pred_counts is None:
            pred_counts = self._build_position_class_counts(pred_masks)
        MI_pred, classwise_mi_pred, Pc_pred = \
            self._mutual_information_from_counts(pred_counts)

        conf_score = MI_pred / self.H_gt_reference \
            if self.H_gt_reference > 0 else 0.0

        H_pred = self._entropy_from_probs(Pc_pred)


        bias_amplification = MI_pred - self.MI_gt

        classwise_bias_amplification = \
            classwise_mi_pred - self.classwise_MI_gt

        active_classes = int(np.sum(Pc_pred > 0))

        return {
            "global_MI_pred": MI_pred,
            "global_NMI_pred": MI_pred/H_pred if H_pred > 0 else 0.0,
            "confounding_score": conf_score,
            "bias_amplification_global": bias_amplification,
            "classwise_MI_pred": classwise_mi_pred,
            "classwise_bias_amplification": classwise_bias_amplification,
            "class_frequencies_pred": Pc_pred,
            "active_classes_pred": active_classes,
        }
    


class PositionStatsAccumulator:
    def __init__(self, n_bins, n_classes, H, W):
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.H = H
        self.W = W

        B = n_bins * n_bins

        # spatial-class counts
        self.counts = np.zeros((B, n_classes), dtype=np.float64)

        # avg/var accumulators
        self.sum_maps = np.zeros((n_classes, H, W), dtype=np.float64)
        self.sq_maps = np.zeros((n_classes, H, W), dtype=np.float64)

        # precompute bin ids
        xs = (np.arange(W) * n_bins // W)
        ys = (np.arange(H) * n_bins // H)
        self.bin_ids = ys[:, None] * n_bins + xs[None, :]

    def update(self, masks):
        """
        masks: (N, H, W)
        """
        masks = np.asarray(masks)

        # GT has -1 for ignore
        # if masks.min() < 0 or masks.max() >= self.n_classes:
        #     raise ValueError(f"masks values must be in [0, {self.n_classes-1}] but got range [{masks.min()}, {masks.max()}].")
        
        N, H, W = masks.shape
        C = self.n_classes
        B = self.n_bins * self.n_bins

        # ===== 1. update mean/var stats =====
        for c in range(C):
            bin_masks = (masks == c).astype(np.float64)

            self.sum_maps[c] += bin_masks.sum(axis=0)
            self.sq_maps[c] += (bin_masks ** 2).sum(axis=0)

        # ===== 2. update spatial-class counts =====
        bins = np.broadcast_to(self.bin_ids, (N, H, W))

        masks = masks.astype(np.int64, copy=False)
        bins = bins.astype(np.int64, copy=False)

        valid = masks != -1   # or masks >= 0

        bins_flat = bins[valid]
        masks_flat = masks[valid]

        # encode (bin, class) pairs into single indices for bincount
        # each (bin, class) pair corresponds to a unique index in [0, B*C-1]
        idx = bins_flat * self.n_classes + masks_flat

        # reshape bincount result back to (B, C) and accumulate
        self.counts += np.bincount(
            idx,
            minlength=self.n_bins * self.n_bins * self.n_classes
        ).reshape(self.n_bins * self.n_bins, self.n_classes)

    def get_avg_var(self, total_N):
        """
        Call after full dataset is processed.
        """
        avg = self.sum_maps / total_N
        var = (self.sq_maps / total_N) - avg**2
        return avg, var

    def get_counts(self):
        return self.counts
    

##########################################################
# Attention MI Analyzer
##########################################################

class AttentionMIAnalyzer:

    def __init__(self, eps=1e-12):
        self.eps = eps

        self.total_samples = 0
        self.head_sum_mi = None
        self.global_mi_sum = 0.0

    def normalize_per_row(self, attn):
        """
        attn: (N, N)
        """

        N = attn.shape[-1]
        row_sums = attn.sum(axis=-1, keepdims=True)
        valid = row_sums.squeeze(-1) > self.eps
        normalized = np.zeros_like(attn)
        normalized[valid] = attn[valid] / row_sums[valid]

        # set fake uniform distribution for rows that sum to 0, so that they don't contribute to MI calculation
        normalized[~valid] = 1.0 / N

        return normalized

    def compute_head_mi(self, head_attn):
        """
        head_attn: (N, N)
        """

        # the case all attention is foused on CLS token, which is a common degenerate case for some heads/layers. 
        # In this case, we consider MI to be 0 since there's no meaningful positional bias.
        if head_attn.sum() == 0:
            return 0.0

        head_attn = self.normalize_per_row(head_attn)

        N = head_attn.shape[0]
        pxy = head_attn / N  # treat each row as a distribution over N positions, and average over rows, so divide by N
        assert np.isclose(pxy.sum(), 1.0, atol=1e-6), f"Expected pxy to sum to 1, got {pxy.sum()}"
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        px_py = px @ py

        mask = pxy > self.eps

        mi = np.sum(
            pxy[mask] * np.log(pxy[mask] / px_py[mask])
        )

        return float(mi)

    def compute_image_mi(self, attn_matrix):
        """
        attn_matrix: (num_heads, N, N)
        """

        num_heads = attn_matrix.shape[0]

        head_mi = np.zeros(num_heads)

        for h in range(num_heads):
            head_mi[h] = self.compute_head_mi(attn_matrix[h])

        return head_mi

    def update(self, batch_attn):
        """
        batch_attn: (B, H, N, N)
        """

        assert batch_attn.ndim == 4, f"Expected batch_attn to have shape (B, H, N, N) but got {batch_attn.shape}"
        B, num_heads, _, _ = batch_attn.shape

        if self.head_sum_mi is None:
            self.head_sum_mi = np.zeros(num_heads)


        # go through each image and compute MI for each head, then accumulate
        for i in range(B):
            img_head_mi = self.compute_image_mi(batch_attn[i])

            self.head_sum_mi += img_head_mi
            self.global_mi_sum += img_head_mi.sum()

        self.total_samples += B

    def finalize(self):

        head_avg = self.head_sum_mi / self.total_samples

        overall_avg = (
            self.global_mi_sum /
            (self.total_samples * len(self.head_sum_mi))
        )

        return {
            "head_avg_mi": head_avg.tolist(),
            "overall_avg_mi": float(overall_avg),
            "num_samples": self.total_samples
        }
