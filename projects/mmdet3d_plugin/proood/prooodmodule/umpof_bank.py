import torch
import torch.nn.functional as F


class UMPOFScorer:
    """Distributional prototype OOD scorer.

    Bank tensors use non-empty class rows. The score is all-class ID
    likelihood converted to OOD energy, then normalized per scene.
    """

    def __init__(self, bank, device='cuda', tau=0.07, eps=1e-6, min_variance=0.02):
        self.device = torch.device(device)
        self.tau = tau
        self.eps = eps

        centers = bank['centers'].float()
        variances = bank['variances'].float()
        priors = bank['priors'].float()

        if centers.dim() != 3:
            raise ValueError('UMPOF centers must have shape [K, M, C].')

        self.class_ids = bank.get('class_ids', None)
        self.metadata = bank.get('metadata', {})
        self.num_classes, self.num_modes, self.feat_dim = centers.shape
        self.tail_class_ids = self._infer_tail_class_ids()

        class_ids = bank.get('class_ids', torch.arange(self.num_classes))
        class_ids = class_ids.long()
        max_class_id = int(class_ids.max().item())
        self.class_lookup = torch.full((max_class_id + 1,), -1, dtype=torch.long)
        for row_idx, class_id in enumerate(class_ids.tolist()):
            self.class_lookup[class_id] = row_idx

        self.centers_by_class = F.normalize(centers, dim=2).to(self.device)
        self.variances_by_class = variances.clamp_min(min_variance).to(self.device)
        self.priors_by_class = priors.clamp_min(eps).to(self.device)
        self.log_priors_by_class = torch.log(self.priors_by_class + eps)
        self.denom_by_class = (
            self.tau * (self.variances_by_class + eps)).clamp_min(eps)
        self.class_lookup = self.class_lookup.to(self.device)
        self.class_ids_tensor = class_ids.to(self.device)

        counts = bank.get('counts', torch.ones_like(variances)).float()
        self.counts_by_class = counts.to(self.device)
        self.class_reliability = self._build_class_reliability(
            counts, variances).to(self.device)
        self.class_tailness = self._build_class_tailness(class_ids).to(self.device)

        centers = centers.reshape(-1, self.feat_dim)
        self.centers = F.normalize(centers, dim=1).to(self.device)
        self.variances = variances.reshape(-1).clamp_min(min_variance).to(self.device)
        self.priors = priors.reshape(-1).clamp_min(eps).to(self.device)
        self.log_priors = torch.log(self.priors + eps)
        self.denom = (self.tau * (self.variances + eps)).clamp_min(eps)

    def _infer_tail_class_ids(self):
        class_names = self.metadata.get('class_names', [])
        if 'other-structure' in class_names or 'other-object' in class_names:
            tail_names = {
                'bicycle', 'motorcycle', 'truck', 'other-object',
                'person', 'pole', 'traffic-sign'
            }
        else:
            tail_names = {
                'bicycle', 'motorcycle', 'truck', 'other-vehicle',
                'person', 'bicyclist', 'motorcyclist', 'pole',
                'traffic-sign', 'other-ground', 'trunk'
            }
        return {
            idx for idx, name in enumerate(class_names)
            if name in tail_names
        }

    def _build_class_reliability(self, counts, variances):
        class_counts = counts.float().sum(dim=1)
        class_variance = variances.float().mean(dim=1)

        count_score = torch.log1p(class_counts)
        count_score = (count_score - count_score.min()) / (
            count_score.max() - count_score.min() + self.eps)

        inv_var = 1.0 / (class_variance + self.eps)
        inv_var = (inv_var - inv_var.min()) / (
            inv_var.max() - inv_var.min() + self.eps)

        reliability = torch.sqrt(torch.clamp(count_score * inv_var, min=0.0))
        return reliability.clamp(0.0, 1.0)

    def _build_class_tailness(self, class_ids):
        tailness = torch.zeros((class_ids.shape[0],), dtype=torch.float32)
        for row_idx, class_id in enumerate(class_ids.tolist()):
            if class_id in self.tail_class_ids:
                tailness[row_idx] = 1.0
        return tailness

    @classmethod
    def from_file(cls, path, device='cuda', tau=0.07, eps=1e-6, min_variance=0.02):
        bank = torch.load(path, map_location='cpu')
        return cls(bank, device=device, tau=tau, eps=eps, min_variance=min_variance)

    @torch.no_grad()
    def score(self, feats, logits=None, pred_labels=None, chunk_size=262144,
              class_conditional=False):
        """Return normalized OOD scores with shape [B, H, W, Z]."""
        if feats.dim() != 5:
            raise ValueError('feats must have shape [B, C, H, W, Z].')

        if feats.shape[1] != self.feat_dim:
            raise ValueError(
                'Feature dim mismatch: bank has {}, input has {}'.format(
                    self.feat_dim, feats.shape[1]))

        if pred_labels is None and logits is not None:
            pred_labels = torch.argmax(logits, dim=1)

        feats = feats.to(self.device)
        if pred_labels is not None:
            pred_labels = pred_labels.to(self.device)

        scores = []
        for batch_idx in range(feats.shape[0]):
            feat_flat = feats[batch_idx].permute(1, 2, 3, 0).reshape(-1, self.feat_dim)
            feat_flat = F.normalize(feat_flat.float(), dim=1)
            if class_conditional:
                if pred_labels is None:
                    raise ValueError('pred_labels or logits are required for class-conditional UMPOF.')
                labels_flat = pred_labels[batch_idx].reshape(-1).long()
                energy_chunks = self._score_class_conditional(
                    feat_flat, labels_flat, chunk_size)
            else:
                energy_chunks = self._score_all_classes(feat_flat, chunk_size)

            energy = torch.cat(energy_chunks, dim=0).reshape(
                feats.shape[2], feats.shape[3], feats.shape[4])
            energy = self._normalize_scene(energy)

            if pred_labels is not None:
                empty_mask = pred_labels[batch_idx] == 0
                if empty_mask.any():
                    energy[empty_mask] = energy.min()

            scores.append(energy)

        return torch.stack(scores, dim=0)

    @torch.no_grad()
    def tadc_gate(self, pred_labels, mode='reliable', tail_boost=0.25,
                  min_gate=0.0, max_gate=1.0):
        if pred_labels.dim() != 4:
            raise ValueError('pred_labels must have shape [B, H, W, Z].')

        pred_labels = pred_labels.to(self.device).long()
        gates = []
        for batch_idx in range(pred_labels.shape[0]):
            labels_flat = pred_labels[batch_idx].reshape(-1)
            gate = torch.zeros((labels_flat.shape[0],), device=self.device)
            valid = (labels_flat >= 0) & (labels_flat < self.class_lookup.shape[0])
            row_indices = torch.full_like(labels_flat, -1)
            row_indices[valid] = self.class_lookup[labels_flat[valid]]
            valid = row_indices >= 0

            if valid.any():
                reliability = self.class_reliability[row_indices[valid]]
                if mode == 'reliable':
                    values = reliability
                elif mode == 'tail':
                    tailness = self.class_tailness[row_indices[valid]]
                    values = reliability * (1.0 + tail_boost * tailness)
                else:
                    raise ValueError('Unsupported TADC gate mode: {}'.format(mode))
                gate[valid] = values.clamp(min_gate, max_gate)

            gates.append(gate.reshape_as(pred_labels[batch_idx]))

        return torch.stack(gates, dim=0)

    @torch.no_grad()
    def tail_mask(self, pred_labels):
        if pred_labels.dim() != 4:
            raise ValueError('pred_labels must have shape [B, H, W, Z].')

        pred_labels = pred_labels.to(self.device).long()
        masks = []
        for batch_idx in range(pred_labels.shape[0]):
            labels_flat = pred_labels[batch_idx].reshape(-1)
            row_indices = torch.full_like(labels_flat, -1)
            valid = (labels_flat >= 0) & (labels_flat < self.class_lookup.shape[0])
            row_indices[valid] = self.class_lookup[labels_flat[valid]]
            valid = row_indices >= 0

            tail = torch.zeros((labels_flat.shape[0],), dtype=torch.bool,
                               device=self.device)
            if valid.any():
                tail[valid] = self.class_tailness[row_indices[valid]] > 0
            masks.append(tail.reshape_as(pred_labels[batch_idx]))

        return torch.stack(masks, dim=0)

    def _score_all_classes(self, feat_flat, chunk_size):
        energy_chunks = []
        for start in range(0, feat_flat.shape[0], chunk_size):
            chunk = feat_flat[start:start + chunk_size]
            cosine = torch.matmul(chunk, self.centers.t())
            dist = 1.0 - cosine
            log_density = self.log_priors.unsqueeze(0) - dist / self.denom.unsqueeze(0)
            id_score = torch.logsumexp(log_density, dim=1)
            energy_chunks.append(-id_score)
        return energy_chunks

    def _score_class_conditional(self, feat_flat, labels_flat, chunk_size):
        energy_chunks = []
        for start in range(0, feat_flat.shape[0], chunk_size):
            chunk = feat_flat[start:start + chunk_size]
            labels = labels_flat[start:start + chunk_size]
            energy = torch.empty((chunk.shape[0],), device=chunk.device)
            valid = (labels >= 0) & (labels < self.class_lookup.shape[0])
            row_indices = torch.full_like(labels, -1)
            row_indices[valid] = self.class_lookup[labels[valid]]
            valid = row_indices >= 0

            if valid.any():
                for row_idx in torch.unique(row_indices[valid]):
                    row_mask = row_indices == row_idx
                    centers = self.centers_by_class[row_idx]
                    cosine = torch.matmul(chunk[row_mask], centers.t())
                    dist = 1.0 - cosine
                    log_density = (
                        self.log_priors_by_class[row_idx].unsqueeze(0)
                        - dist / self.denom_by_class[row_idx].unsqueeze(0))
                    id_score = torch.logsumexp(log_density, dim=1)
                    energy[row_mask] = -id_score
                energy[~valid] = energy[valid].min()
            else:
                energy.zero_()

            energy_chunks.append(energy)
        return energy_chunks

    def _normalize_scene(self, score):
        min_val = score.min()
        max_val = score.max()
        if max_val > min_val:
            return (score - min_val) / (max_val - min_val)
        return torch.zeros_like(score)
