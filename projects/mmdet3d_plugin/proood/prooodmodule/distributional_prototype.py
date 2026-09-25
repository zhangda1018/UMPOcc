import torch
import torch.nn.functional as F

from .prototype import PrototypeModule


class DistributionalPrototypeModule(PrototypeModule):
    """EMA mixture prototypes with the original class-level compatibility API.

    ``prototypes`` remains a [num_classes, C] class mean so existing PGSI and
    PGTM code keeps working. ``sub_prototypes`` stores the M-mode distribution
    used by the contrastive loss and feature enhancement.
    """

    def __init__(self, *args, num_modes=4, assignment_temperature=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if num_modes < 2:
            raise ValueError('Distributional prototypes require at least 2 modes.')
        self.num_modes = int(num_modes)
        self.assignment_temperature = float(assignment_temperature)
        self.register_buffer(
            'sub_prototypes',
            torch.zeros(self.num_prototype_classes, self.num_modes,
                        self.prototype_dim))
        self.register_buffer(
            'sub_initialized',
            torch.zeros(self.num_prototype_classes, self.num_modes).bool())
        self.register_buffer(
            'sub_quality',
            torch.zeros(self.num_prototype_classes, self.num_modes))
        self.register_buffer(
            'sub_priors',
            torch.full((self.num_prototype_classes, self.num_modes),
                       1.0 / self.num_modes))

    def _feature_sample(self, features, max_features=4096):
        if features.shape[0] <= max_features:
            return features
        indices = torch.linspace(
            0, features.shape[0] - 1, max_features,
            device=features.device).long()
        return features[indices]

    @torch.no_grad()
    def update_prototypes(self, vox_feats_full, target):
        B, C, _, _, _ = vox_feats_full.shape
        feats_flat = vox_feats_full.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        target_flat = target.reshape(B, -1)

        for batch_idx in range(B):
            for cls_id in torch.unique(target_flat[batch_idx]).tolist():
                if cls_id in (0, 255) or cls_id not in self.prototype_class_mapping:
                    continue
                proto_idx = self.prototype_class_mapping[cls_id]
                cls_feats = self._feature_sample(
                    feats_flat[batch_idx][target_flat[batch_idx] == cls_id])
                if cls_feats.shape[0] == 0:
                    continue

                if not self.prototype_initialized[proto_idx]:
                    means = []
                    counts = []
                    for mode_idx in range(self.num_modes):
                        chunk = cls_feats[mode_idx::self.num_modes]
                        if chunk.shape[0] == 0:
                            means.append(cls_feats.mean(dim=0))
                            counts.append(0.0)
                        else:
                            means.append(chunk.mean(dim=0))
                            counts.append(float(chunk.shape[0]))
                    means = torch.stack(means, dim=0)
                    counts = torch.tensor(counts, device=means.device)
                    if counts.sum() == 0:
                        counts.fill_(1.0)
                    self.sub_prototypes[proto_idx].copy_(means)
                    self.sub_initialized[proto_idx].fill_(True)
                    self.sub_priors[proto_idx].copy_(counts / counts.sum())
                    self.prototypes[proto_idx].copy_(
                        (self.sub_priors[proto_idx, :, None] * means).sum(dim=0))
                    self.prototype_initialized[proto_idx] = True
                    self.prototype_quality[proto_idx] = 1.0
                    self.sub_quality[proto_idx].fill_(1.0)
                    continue

                current = self.sub_prototypes[proto_idx]
                feat_norm = F.normalize(cls_feats, dim=1)
                proto_norm = F.normalize(current, dim=1)
                distances = 1.0 - feat_norm @ proto_norm.t()
                assignments = F.softmax(
                    -distances / self.assignment_temperature, dim=1)
                mass = assignments.sum(dim=0)
                for mode_idx in range(self.num_modes):
                    if mass[mode_idx] <= 1e-5:
                        continue
                    mean = (assignments[:, mode_idx, None] * cls_feats).sum(dim=0)
                    mean = mean / mass[mode_idx]
                    momentum = self.ema_momentum
                    self.sub_prototypes[proto_idx, mode_idx].mul_(1 - momentum)
                    self.sub_prototypes[proto_idx, mode_idx].add_(momentum * mean)
                    variance = (
                        assignments[:, mode_idx] * distances[:, mode_idx]).sum()
                    variance = variance / mass[mode_idx]
                    self.sub_quality[proto_idx, mode_idx].mul_(1 - momentum)
                    self.sub_quality[proto_idx, mode_idx].add_(
                        momentum / (variance + 1e-5))

                priors = mass / mass.sum().clamp_min(1e-6)
                self.sub_priors[proto_idx].mul_(1 - self.ema_momentum)
                self.sub_priors[proto_idx].add_(self.ema_momentum * priors)
                self.prototypes[proto_idx].copy_(
                    (self.sub_priors[proto_idx, :, None] *
                     self.sub_prototypes[proto_idx]).sum(dim=0))
                self.prototype_quality[proto_idx].copy_(
                    self.sub_quality[proto_idx].mean())

    def prototype_contrastive_loss(self, feats_flat, target_flat, temperature=0.1):
        valid_mask = (target_flat != 255) & (target_flat != 0)
        if not valid_mask.any():
            return feats_flat.sum() * 0.0

        feats = self._feature_sample(feats_flat[valid_mask])
        labels = target_flat[valid_mask]
        if feats.shape[0] != labels.shape[0]:
            labels = labels[torch.linspace(
                0, labels.shape[0] - 1, feats.shape[0],
                device=labels.device).long()]

        prototypes = F.normalize(
            self.sub_prototypes.reshape(-1, self.prototype_dim), dim=1)
        features = F.normalize(feats, dim=1)
        mode_logits = features @ prototypes.t() / temperature
        mode_logits = mode_logits.reshape(
            features.shape[0], self.num_prototype_classes, self.num_modes)
        mode_logits = mode_logits + torch.log(
            self.sub_priors.clamp_min(1e-6))[None]
        class_logits = torch.logsumexp(mode_logits, dim=2)

        label_to_index = torch.full(
            (256,), -1, device=labels.device, dtype=torch.long)
        for cls_id, proto_idx in self.prototype_class_mapping.items():
            label_to_index[cls_id] = proto_idx
        proto_labels = label_to_index[labels.long()]
        valid = proto_labels >= 0
        if not valid.any():
            return feats_flat.sum() * 0.0
        rows = torch.arange(features.shape[0], device=features.device)[valid]
        positive = class_logits[rows, proto_labels[valid]]
        denominator = torch.logsumexp(class_logits[rows], dim=1)
        return (denominator - positive).mean()

    def tail_aware_boundary_loss(self, feats_flat, target_flat,
                                 tail_class_ids, margin=0.15,
                                 tail_margin_boost=0.5, tail_weight=1.0,
                                 tail_only=False,
                                 max_features=4096):
        """Separate each ID feature from its closest competing class mode.

        The loss is supervised by occupancy labels and uses the same mixture
        modes as UMPOF. Tail classes receive a wider margin and a higher
        contribution, which calibrates the learned density boundary without
        introducing synthetic OOD labels.
        """
        valid = (target_flat != 255) & (target_flat != 0)
        if not valid.any() or not self.sub_initialized.any():
            return feats_flat.sum() * 0.0

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        valid_labels = target_flat[valid_indices].long()
        present_labels = [
            int(cls_id) for cls_id in torch.unique(valid_labels).tolist()
            if int(cls_id) in self.prototype_class_mapping
        ]
        if not present_labels:
            return feats_flat.sum() * 0.0

        # Preserve rare and tail classes instead of allowing road/building
        # voxels to dominate the boundary objective.
        per_class = max(1, int(max_features) // len(present_labels))
        sampled_indices = []
        for cls_id in present_labels:
            cls_indices = valid_indices[valid_labels == cls_id]
            if cls_indices.numel() > per_class:
                positions = torch.linspace(
                    0, cls_indices.numel() - 1, per_class,
                    device=cls_indices.device).long()
                cls_indices = cls_indices[positions]
            sampled_indices.append(cls_indices)
        indices = torch.cat(sampled_indices, dim=0)

        feats = F.normalize(feats_flat[indices].float(), dim=1)
        labels = target_flat[indices].long()
        label_to_index = torch.full(
            (256,), -1, device=labels.device, dtype=torch.long)
        for cls_id, proto_idx in self.prototype_class_mapping.items():
            label_to_index[cls_id] = proto_idx
        class_indices = label_to_index[labels]
        valid_classes = class_indices >= 0
        if not valid_classes.any():
            return feats_flat.sum() * 0.0

        feats = feats[valid_classes]
        labels = labels[valid_classes]
        class_indices = class_indices[valid_classes]
        tail_ids = torch.tensor(
            list(tail_class_ids), device=labels.device, dtype=labels.dtype)
        is_tail = (labels[:, None] == tail_ids[None]).any(dim=1)
        if tail_only:
            if not is_tail.any():
                return feats_flat.sum() * 0.0
            feats = feats[is_tail]
            labels = labels[is_tail]
            class_indices = class_indices[is_tail]
            is_tail = is_tail[is_tail]

        prototypes = F.normalize(self.sub_prototypes.float(), dim=2)
        distances = 1.0 - torch.einsum('nc,kmc->nkm', feats, prototypes)
        distances = distances.masked_fill(
            ~self.sub_initialized[None], float('inf'))

        rows = torch.arange(distances.shape[0], device=distances.device)
        positive = distances[rows, class_indices].min(dim=1).values
        rival_distances = distances.clone()
        rival_distances[rows, class_indices] = float('inf')
        rival = rival_distances.reshape(distances.shape[0], -1).min(dim=1).values
        finite = torch.isfinite(positive) & torch.isfinite(rival)
        if not finite.any():
            return feats_flat.sum() * 0.0

        margins = torch.full_like(positive, float(margin))
        margins = margins * (1.0 + float(tail_margin_boost) * is_tail.float())
        weights = 1.0 + float(tail_weight) * is_tail.float()
        loss = F.relu(margins + positive - rival)
        return (loss[finite] * weights[finite]).sum() / weights[finite].sum().clamp_min(1.0)

    def enhance_features(self, vox_feats_diff, masked_idx, unmasked_idx,
                          aux_occ_logit=None):
        if not self.use_ema or not self.should_use_enhancement():
            return vox_feats_diff
        if not self.sub_initialized.any():
            return vox_feats_diff

        B, C, H, W, Z = vox_feats_diff.shape
        N = H * W * Z
        valid = self.sub_initialized.reshape(-1)
        prototypes = self.sub_prototypes.reshape(-1, C)[valid]
        prototypes = prototypes.to(vox_feats_diff.dtype)
        prototypes_norm = F.normalize(prototypes, dim=1)
        feats = vox_feats_diff.reshape(B, C, N)
        features_norm = F.normalize(feats, dim=1)
        similarity = torch.einsum(
            'bcn,kc->bkn', features_norm, prototypes_norm)
        weights = F.softmax(similarity / self.temp, dim=1)
        enhanced = torch.einsum('bkn,kc->bnc', weights, prototypes)

        mask = torch.zeros(N, dtype=torch.bool, device=vox_feats_diff.device)
        mask[torch.from_numpy(masked_idx[0]).long().to(mask.device)] = True
        if aux_occ_logit is None:
            pred_nonempty = torch.ones(B, N, dtype=torch.bool, device=mask.device)
        else:
            pred_nonempty = (aux_occ_logit.argmax(dim=1) != 0).reshape(B, N)
        enhance_mask = mask[None] & pred_nonempty
        final = feats.permute(0, 2, 1).clone()
        if enhance_mask.any():
            final[enhance_mask] += self.alpha_scalar * enhanced[enhance_mask]
        return final.permute(0, 2, 1).reshape(B, C, H, W, Z)
