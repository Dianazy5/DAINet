import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BertModel, CLIPModel

import models_mae
from utils.utils_finefake import Averager, Recorder, calculate_metrics, clipdata2gpu

logger = logging.getLogger(__name__)
















class BinaryFocalWithLogitsLoss(nn.Module):


    def __init__(self, gamma=1.5, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class FeedForwardMLP(nn.Module):


    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, use_batchnorm=True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiScaleProjector(nn.Module):


    def __init__(self, in_dim, out_dim, num_scales, dropout=0.1, lazy=False):
        super().__init__()
        self.num_scales = num_scales
        linear_cls = nn.LazyLinear if lazy else nn.Linear

        self.scale_layers = nn.ModuleList(
            [linear_cls(out_dim) if lazy else linear_cls(in_dim, out_dim) for _ in range(num_scales)])

        gate_hidden = max(out_dim, 128)
        gate_bottleneck = max(out_dim // 2, 64)
        gate_first = linear_cls(gate_hidden) if lazy else nn.Linear(in_dim, gate_hidden)
        self.scale_gate = nn.Sequential(
            gate_first,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, gate_bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_bottleneck, num_scales),
        )

        self.fuse = nn.Sequential(
            nn.Linear(out_dim * num_scales, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        if (not lazy) and in_dim == out_dim:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = linear_cls(out_dim) if lazy else nn.Linear(in_dim, out_dim)

        self.out_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        scale_features = [self.dropout(F.gelu(layer(x))) for layer in self.scale_layers]

        pooled_x = x.mean(dim=1) if x.dim() == 3 else x
        scale_weights = torch.softmax(self.scale_gate(pooled_x), dim=-1)

        if x.dim() == 3:
            stacked = torch.stack(scale_features, dim=2)
            weighted = stacked * scale_weights[:, None, :, None]
            fused_in = weighted.reshape(x.size(0), x.size(1), -1)
        else:
            stacked = torch.stack(scale_features, dim=1)
            weighted = stacked * scale_weights[:, :, None]
            fused_in = weighted.reshape(x.size(0), -1)

        fused = self.fuse(fused_in)
        residual = self.residual_proj(x)
        return self.out_norm(residual + fused)


class LocalLocalInconsistency(nn.Module):


    def __init__(self, hidden_dim, dropout=0.1, num_heads=8, out_channels=64, kernel_size=3):
        super().__init__()
        num_heads = max(1, min(num_heads, hidden_dim))
        while hidden_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads

        self.text_to_image_attn = nn.MultiheadAttention(
            hidden_dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            hidden_dim,
            self.num_heads,
            dropout=dropout,
            batch_first=True,
        )

        mlp_hidden = hidden_dim * 2
        self.inconsistency_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, F_T, F_I):
        F_I_given_T, _ = self.text_to_image_attn(
            query=F_T,
            key=F_I,
            value=F_I,
            need_weights=False,
        )
        F_T_given_I, _ = self.image_to_text_attn(
            query=F_I,
            key=F_T,
            value=F_T,
            need_weights=False,
        )

        IC_T_to_I = self.inconsistency_mlp(
            torch.cat([F_T, F_I_given_T, F_T - F_I_given_T, F_T * F_I_given_T], dim=-1)
        )
        IC_I_to_T = self.inconsistency_mlp(
            torch.cat([F_I, F_T_given_I, F_I - F_T_given_I, F_I * F_T_given_I], dim=-1)
        )

        pooled_ti = IC_T_to_I.mean(dim=1)
        pooled_it = IC_I_to_T.mean(dim=1)
        return self.fusion(torch.cat([pooled_ti, pooled_it], dim=-1))


class GlobalLocalInconsistency(nn.Module):


    def __init__(self, hidden_dim, dropout=0.1, num_heads=8):
        super().__init__()
        num_heads = max(1, min(num_heads, hidden_dim))
        while hidden_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)

        self.resid_dropout = nn.Dropout(dropout)
        self.norm_t1 = nn.LayerNorm(hidden_dim)
        self.norm_i1 = nn.LayerNorm(hidden_dim)
        self.norm_t2 = nn.LayerNorm(hidden_dim)
        self.norm_i2 = nn.LayerNorm(hidden_dim)

        ff_dim = hidden_dim * 3
        self.ffn_t = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.entropy_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.interaction_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, text_global, image_global, text_local, image_local):
        text_query0 = text_global.unsqueeze(1)
        image_query0 = image_global.unsqueeze(1)

        text_cross, text_attn = self.text_to_image(text_query0, image_local, image_local)
        image_cross, image_attn = self.image_to_text(image_query0, text_local, text_local)
        text_query1 = self.norm_t1(text_query0 + self.resid_dropout(text_cross))
        image_query1 = self.norm_i1(image_query0 + self.resid_dropout(image_cross))

        text_query2 = self.norm_t2(text_query1 + self.ffn_t(text_query1))
        image_query2 = self.norm_i2(image_query1 + self.ffn_i(image_query1))

        text_refined = text_query2.squeeze(1)
        image_refined = image_query2.squeeze(1)

        text_gl_offset = text_refined - text_global
        image_gl_offset = image_refined - image_global
        cross_offset = text_refined - image_refined

        text_attn = text_attn.squeeze(1).clamp_min(1e-8)
        image_attn = image_attn.squeeze(1).clamp_min(1e-8)
        text_entropy = -(text_attn * text_attn.log()).sum(dim=-1, keepdim=True)
        image_entropy = -(image_attn * image_attn.log()).sum(dim=-1, keepdim=True)
        entropy_feat = self.entropy_proj(torch.cat([text_entropy, image_entropy], dim=-1))

        interaction = torch.cat([text_gl_offset, image_gl_offset, cross_offset, entropy_feat], dim=-1)
        interaction = interaction * self.interaction_gate(interaction)
        return self.out(interaction)


class HierarchicalConflictSynergyNetwork(nn.Module):


    def __init__(self, dim_ll, dim_gl, dim_gg, feature_dim=512, num_heads=8, dropout=0.1, with_classifier=False):
        super().__init__()

        self.proj_ll = nn.Identity() if dim_ll == feature_dim else nn.Linear(dim_ll, feature_dim)
        self.proj_gl = nn.Identity() if dim_gl == feature_dim else nn.Linear(dim_gl, feature_dim)
        self.proj_gg = nn.Identity() if dim_gg == feature_dim else nn.Linear(dim_gg, feature_dim)

        num_heads = max(1, min(num_heads, feature_dim))
        while feature_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.node_type_embed = nn.Parameter(torch.randn(3, feature_dim) * 0.02)
        self.node_refine = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.post_norm = nn.LayerNorm(feature_dim)

        self.with_classifier = with_classifier
        self.classifier = (
            nn.Sequential(
                nn.Linear(feature_dim * 3, max(feature_dim // 2, 128)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(feature_dim // 2, 128), 1),
            )
            if with_classifier
            else None
        )

    def forward(self, c_ll, c_gl, c_gg):
        c_ll_proj = self.proj_ll(c_ll)
        c_gl_proj = self.proj_gl(c_gl)
        c_gg_proj = self.proj_gg(c_gg)

        conflict_nodes = torch.stack([c_ll_proj, c_gl_proj, c_gg_proj], dim=1)
        conflict_nodes = conflict_nodes + self.node_type_embed.unsqueeze(0)

        synergy_nodes = self.transformer_block(conflict_nodes)
        synergy_nodes = self.post_norm(conflict_nodes + self.node_refine(synergy_nodes))

        c_ll_hat = synergy_nodes[:, 0, :]
        c_gl_hat = synergy_nodes[:, 1, :]
        c_gg_hat = synergy_nodes[:, 2, :]

        fake_score = None
        if self.with_classifier and self.classifier is not None:
            final_feature = torch.cat([c_ll_hat, c_gl_hat, c_gg_hat], dim=-1)
            fake_score = self.classifier(final_feature).squeeze(-1)

        return fake_score, (c_ll_hat, c_gl_hat, c_gg_hat)


class MultiDomainD2IANModel(nn.Module):


    def __init__(
        self,
        emb_dim,
        mlp_dims,
        bert_path_or_name,
        clip_path_or_name,
        out_channels,
        dropout,
        use_cuda=True,
        domain_num=2,
        num_scales=4,
        unknown_domain_id=None,
    ):
        super().__init__()

        self.hidden_dim = int(out_channels) if out_channels is not None and int(out_channels) > 0 else emb_dim
        self.domain_num = max(int(domain_num), 1)
        self.num_scales = max(int(num_scales), 1)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.unknown_domain_id = int(unknown_domain_id) if unknown_domain_id is not None else None


        self.text_local_input_dim = 768
        self.image_local_input_dim = 768
        self.clip_output_dim = 512


        self.bert = self._build_bert(bert_path_or_name)
        self.image_model = self._build_mae_model()
        self.clip_model = self._build_clip_model(clip_path_or_name)


        self.text_local_proj = MultiScaleProjector(self.text_local_input_dim, self.hidden_dim, self.num_scales, dropout=dropout)
        self.image_local_proj = MultiScaleProjector(self.image_local_input_dim, self.hidden_dim, self.num_scales, dropout=dropout)
        self.text_global_proj = MultiScaleProjector(self.clip_output_dim, self.hidden_dim, self.num_scales, dropout=dropout)
        self.image_global_proj = MultiScaleProjector(self.clip_output_dim, self.hidden_dim, self.num_scales, dropout=dropout)


        self.gg_branch = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.ll_branch = LocalLocalInconsistency(self.hidden_dim, dropout=dropout)
        self.gl_branch = GlobalLocalInconsistency(self.hidden_dim, dropout=dropout)


        self.conflict_synergy = HierarchicalConflictSynergyNetwork(
            dim_ll=self.hidden_dim,
            dim_gl=self.hidden_dim,
            dim_gg=self.hidden_dim,
            feature_dim=self.hidden_dim,
            num_heads=8,
            dropout=dropout,
            with_classifier=False,
        )


        self.domain_embedding = nn.Embedding(self.domain_num, self.hidden_dim)
        self.domain_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.domain_num),
        )
        self.domain_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 3),
        )
        self.conflict_prefer_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 3),
        )
        self.context_residual = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.context_residual_norm = nn.LayerNorm(self.hidden_dim)


        self.domain_recalibration = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 3),
        )
        self.final_classifier = FeedForwardMLP(
            input_dim=self.hidden_dim * 3,
            hidden_dims=[self.hidden_dim * 2, self.hidden_dim],
            output_dim=1,
            dropout=dropout,
            use_batchnorm=True,
        )

        aux_hidden = self.hidden_dim // 2
        self.ll_classifier = FeedForwardMLP(
            input_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim, aux_hidden],
            output_dim=1,
            dropout=dropout,
            use_batchnorm=True,
        )
        self.gl_classifier = FeedForwardMLP(
            input_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim, aux_hidden],
            output_dim=1,
            dropout=dropout,
            use_batchnorm=True,
        )
        self.gg_classifier = FeedForwardMLP(
            input_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim, aux_hidden],
            output_dim=1,
            dropout=dropout,
            use_batchnorm=True,
        )

    def _build_bert(self, bert_path_or_name):
        logger.info(f"Loading BERT: {bert_path_or_name}")
        try:
            model = BertModel.from_pretrained(bert_path_or_name, local_files_only=True)
            logger.info("BERT loaded from local files.")
        except Exception as local_err:
            logger.warning(f"Local BERT load failed, trying fallback path. Details: {local_err}")
            try:
                model = BertModel.from_pretrained(bert_path_or_name)
                logger.info("BERT loaded from fallback path.")
            except Exception as exc:
                logger.exception(f"Failed to load BERT from {bert_path_or_name}: {exc}")
                return None
        model.requires_grad_(False)
        return model

    def _build_mae_model(self):
        model_size = "base"
        ckpt_path = f"./mae_pretrain_vit_{model_size}.pth"
        try:
            mae_model = models_mae.__dict__[f"mae_vit_{model_size}_patch16"](norm_pix_loss=False)
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                mae_model.load_state_dict(state_dict, strict=False)
                logger.info(f"MAE loaded from: {ckpt_path}")
            else:
                logger.warning(f"MAE checkpoint not found: {ckpt_path}. Using randomly initialized MAE backbone.")
            mae_model.requires_grad_(False)
            return mae_model
        except Exception as exc:
            logger.exception(f"Failed to build/load MAE: {exc}")
            return None

    def _build_clip_model(self, clip_path_or_name):
        logger.info(f"Loading CLIP: {clip_path_or_name}")
        try:
            model = CLIPModel.from_pretrained(clip_path_or_name, local_files_only=True)
            logger.info("CLIP loaded from local files.")
        except Exception as local_err:
            logger.warning(f"Local CLIP load failed, trying fallback path. Details: {local_err}")
            try:
                model = CLIPModel.from_pretrained(clip_path_or_name)
                logger.info("CLIP loaded from fallback path.")
            except Exception as exc:
                logger.exception(f"Failed to load CLIP from {clip_path_or_name}: {exc}")
                return None
        model.requires_grad_(False)
        return model

    def _encode_image_local(self, image_raw):
        if self.image_model is None:
            return None
        try:
            image_local = self.image_model.forward_ying(image_raw)
        except AttributeError:
            image_local = self.image_model.forward_features(image_raw)
        except Exception as exc:
            logger.warning(f"MAE local encoding failed: {exc}")
            return None

        if image_local.dim() == 2:
            image_local = image_local.unsqueeze(1)
        return image_local

    def _encode_clip_global(self, clip_image_input, clip_text_input, clip_attention_mask, batch_size, device, dtype):

        zeros_global = torch.zeros(batch_size, self.clip_output_dim, device=device, dtype=dtype)

        if self.clip_model is None or clip_image_input is None or clip_text_input is None:
            return zeros_global, zeros_global

        try:
            with torch.no_grad():
                image_global = self.clip_model.get_image_features(pixel_values=clip_image_input)
                text_global = self.clip_model.get_text_features(
                    input_ids=clip_text_input,
                    attention_mask=clip_attention_mask,
                )
                image_global = image_global / image_global.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                text_global = text_global / text_global.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        except Exception as exc:
            logger.warning(f"CLIP global encoding failed, fallback to zeros. Details: {exc}")
            return zeros_global, zeros_global


        if (
            image_global.dim() != 2
            or text_global.dim() != 2
            or image_global.size(-1) != self.clip_output_dim
            or text_global.size(-1) != self.clip_output_dim
        ):
            logger.warning(
                "CLIP output shape mismatch "
                f"(image={tuple(image_global.shape)}, text={tuple(text_global.shape)}), fallback to zeros."
            )
            return zeros_global, zeros_global

        return text_global.float(), image_global.float()

    def forward(self, **kwargs):
        inputs = kwargs["content"]
        masks = kwargs["content_masks"]
        image_raw = kwargs["image"]
        clip_image_input = kwargs.get("clip_image")
        clip_text_input = kwargs.get("clip_text")
        clip_attention_mask = kwargs.get("clip_attention_mask")


        if self.bert is not None:
            try:
                text_local_raw = self.bert(input_ids=inputs, attention_mask=masks).last_hidden_state
            except Exception as exc:
                logger.warning(f"BERT forward failed, fallback to zero local text feature. Details: {exc}")
                text_local_raw = torch.zeros(
                    inputs.size(0),
                    inputs.size(1),
                    self.text_local_input_dim,
                    device=inputs.device,
                    dtype=torch.float32,
                )
        else:
            text_local_raw = torch.zeros(
                inputs.size(0),
                inputs.size(1),
                self.text_local_input_dim,
                device=inputs.device,
                dtype=torch.float32,
            )


        image_local_raw = self._encode_image_local(image_raw)
        if image_local_raw is None:
            image_local_raw = torch.zeros_like(text_local_raw)


        text_local = self.text_local_proj(text_local_raw)
        image_local = self.image_local_proj(image_local_raw)

        text_global_raw, image_global_raw = self._encode_clip_global(
            clip_image_input,
            clip_text_input,
            clip_attention_mask,
            batch_size=inputs.size(0),
            device=inputs.device,
            dtype=text_local_raw.dtype,
        )
        text_global = self.text_global_proj(text_global_raw)
        image_global = self.image_global_proj(image_global_raw)


        conflict_gg = self.gg_branch(torch.cat([text_global, image_global, text_global - image_global], dim=-1))

        conflict_ll = self.ll_branch(text_local, image_local)

        conflict_gl = self.gl_branch(text_global, image_global, text_local, image_local)


        _, (conflict_ll_hat, conflict_gl_hat, conflict_gg_hat) = self.conflict_synergy(
            conflict_ll, conflict_gl, conflict_gg
        )


        global_context = torch.cat([text_global, image_global], dim=-1)
        pred_domain_logits = self.domain_predictor(global_context)
        if self.unknown_domain_id is not None and 0 <= self.unknown_domain_id < self.domain_num and self.domain_num > 1:
            pred_domain_logits = pred_domain_logits.clone()
            pred_domain_logits[:, self.unknown_domain_id] = torch.finfo(pred_domain_logits.dtype).min
        pred_domain_prob = torch.softmax(pred_domain_logits, dim=-1)
        pred_domain_vec = torch.matmul(pred_domain_prob, self.domain_embedding.weight)

        category = kwargs.get("category")
        if category is None:
            domain_vec = pred_domain_vec
        else:
            domain_ids = category.long()
            use_predicted_domain = (domain_ids < 0) | (domain_ids >= self.domain_num)
            if self.unknown_domain_id is not None:
                use_predicted_domain = use_predicted_domain | (domain_ids == self.unknown_domain_id)
            domain_ids = domain_ids.clamp(min=0, max=self.domain_num - 1)
            label_domain_vec = self.domain_embedding(domain_ids)
            mixed_label_domain_vec = 0.7 * label_domain_vec + 0.3 * pred_domain_vec
            domain_vec = torch.where(use_predicted_domain.unsqueeze(-1), pred_domain_vec, mixed_label_domain_vec)


        domain_weights = torch.softmax(self.domain_gate(domain_vec), dim=-1)
        conflict_prefer = torch.softmax(self.conflict_prefer_gate(global_context), dim=-1)
        domain_weights = torch.softmax(
            torch.log(domain_weights.clamp_min(1e-6)) + torch.log(conflict_prefer.clamp_min(1e-6)),
            dim=-1,
        )


        conflict_fused = (
            domain_weights[:, 0:1] * conflict_ll_hat
            + domain_weights[:, 1:2] * conflict_gl_hat
            + domain_weights[:, 2:3] * conflict_gg_hat
        )

        domain_unit = F.normalize(domain_vec, dim=-1)
        proj_scalar = (conflict_fused * domain_unit).sum(dim=-1, keepdim=True)
        conflict_orth = conflict_fused - proj_scalar * domain_unit


        ortho_penalty = (F.cosine_similarity(conflict_fused, domain_vec, dim=-1) ** 2).mean()

        context_residual = self.context_residual(global_context)
        adaptive_conflict = self.context_residual_norm(conflict_orth + context_residual)


        fusion_feature = torch.cat([text_global, image_global, adaptive_conflict], dim=-1)
        recalibration = torch.sigmoid(self.domain_recalibration(domain_vec))
        fusion_feature = fusion_feature * recalibration


        final_logits = self.final_classifier(fusion_feature).squeeze(1)
        ll_logits = self.ll_classifier(conflict_ll_hat).squeeze(1)
        gl_logits = self.gl_classifier(conflict_gl_hat).squeeze(1)
        gg_logits = self.gg_classifier(conflict_gg_hat).squeeze(1)

        return final_logits, ll_logits, gl_logits, gg_logits, adaptive_conflict, domain_weights, ortho_penalty


class Trainer:


    def __init__(
        self,
        emb_dim,
        mlp_dims,
        bert_path_or_name,
        clip_path_or_name,
        use_cuda,
        lr,
        dropout,
        train_loader,
        val_loader,
        test_loader,
        category_dict,
        weight_decay,
        save_param_dir,
        early_stop=10,
        epoches=100,
        metric_key_for_early_stop="metric",
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.save_param_dir = save_param_dir
        self.metric_key_for_early_stop = metric_key_for_early_stop
        os.makedirs(self.save_param_dir, exist_ok=True)

        domain_num = len(self.category_dict) if isinstance(self.category_dict, dict) and self.category_dict else 1
        unknown_domain_id = self.category_dict.get("Uncategorized") if isinstance(self.category_dict, dict) else None
        self.model = MultiDomainD2IANModel(
            emb_dim=self.emb_dim,
            mlp_dims=self.mlp_dims,
            bert_path_or_name=bert_path_or_name,
            clip_path_or_name=clip_path_or_name,
            out_channels=self.mlp_dims[0] if isinstance(self.mlp_dims, (list, tuple)) and self.mlp_dims else 320,
            dropout=self.dropout,
            use_cuda=self.use_cuda,
            domain_num=domain_num,
            num_scales=4,
            unknown_domain_id=unknown_domain_id,
        ).to(self.device)


        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = BinaryFocalWithLogitsLoss(gamma=1.5)
        self.focal_main_weight = 0.35


        self.aux_weight_start = 0.45
        self.aux_weight_end = 0.15


        self.consistency_weight = 0.05


        self.ortho_weight_start = 0.005
        self.ortho_weight_end = 0.03

        self.label_smoothing = 0.02
        self.max_grad_norm = 5.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epoches,
            eta_min=self.lr * 0.2,
        )
        self.model_save_filename = "parameter_d2ian_finefake.pkl"

    def _prepare_batch(self, batch):
        if batch is None:
            logger.warning("Received None batch.")
            return None

        return clipdata2gpu(batch, use_cuda=self.use_cuda, device=self.device)

    def train(self):
        recorder = Recorder(self.early_stop, metric_key=self.metric_key_for_early_stop)
        logger.info(
            f"Training start: epochs={self.epoches}, early_stop={self.early_stop}, "
            f"metric={self.metric_key_for_early_stop}, lr={self.lr}, weight_decay={self.weight_decay}"
        )

        for epoch in range(self.epoches):
            self.model.train()
            train_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_iter):
                try:
                    batch_data = self._prepare_batch(batch)
                    if batch_data is None:
                        logger.warning(f"Skipping batch {step_n} due to data loading/GPU transfer error.")
                        continue
                    if "label" not in batch_data:
                        logger.warning(f"Skipping batch {step_n} due to missing label.")
                        continue

                    labels = batch_data["label"].float().view(-1)
                    labels_train = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
                    final_logits, ll_logits, gl_logits, gg_logits, _, _, ortho_penalty = self.model(**batch_data)

                    epoch_ratio = epoch / max(1, self.epoches - 1)
                    aux_weight = self.aux_weight_start + (self.aux_weight_end - self.aux_weight_start) * epoch_ratio
                    ortho_weight = self.ortho_weight_start + (self.ortho_weight_end - self.ortho_weight_start) * epoch_ratio


                    loss_main_bce = self.bce_loss(final_logits, labels_train)
                    loss_main_focal = self.focal_loss(final_logits, labels_train)
                    loss_main = loss_main_bce + self.focal_main_weight * loss_main_focal


                    loss_aux = (
                        self.bce_loss(ll_logits, labels_train)
                        + self.bce_loss(gl_logits, labels_train)
                        + self.bce_loss(gg_logits, labels_train)
                    ) / 3.0


                    with torch.no_grad():
                        teacher_prob = torch.sigmoid(final_logits)
                    loss_consistency = (
                        F.smooth_l1_loss(torch.sigmoid(ll_logits), teacher_prob)
                        + F.smooth_l1_loss(torch.sigmoid(gl_logits), teacher_prob)
                        + F.smooth_l1_loss(torch.sigmoid(gg_logits), teacher_prob)
                    ) / 3.0


                    total_loss = (
                        loss_main
                        + aux_weight * loss_aux
                        + self.consistency_weight * loss_consistency
                        + ortho_weight * ortho_penalty
                    )

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()

                    avg_loss.add(total_loss.item())
                    train_iter.set_description(f"Epoch {epoch + 1}/{self.epoches}")
                    train_iter.set_postfix(
                        loss=avg_loss.item(),
                        main=loss_main.item(),
                        aux=loss_aux.item(),
                        ortho=ortho_penalty.item(),
                        w_aux=aux_weight,
                        w_ortho=ortho_weight,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if "must match the size" in err_msg:
                        logger.error(f"Tensor mismatch error at Train step {epoch + 1}-{step_n}: {exc}", exc_info=True)
                    elif "image" in err_msg or "channel" in err_msg or "collate_fn" in err_msg:
                        logger.error(f"Image processing related error at Train step {epoch + 1}-{step_n}: {exc}", exc_info=True)
                    else:
                        logger.exception(f"Train step {step_n} error: {exc}")
                    continue

            logger.info(
                f"Train Epoch {epoch + 1} Done; Avg Loss: {avg_loss.item():.4f}; "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if self.val_loader is None:
                logger.warning("Val loader not provided, skipping validation.")
                continue

            try:
                val_results = self.test(self.val_loader)
                if not val_results:
                    logger.warning(f"Val epoch {epoch + 1} returned empty/invalid results. Skipping this epoch.")
                    continue

                tracked_metric = val_results.get(self.metric_key_for_early_stop, 0.0)
                acc_val = val_results.get("acc", 0.0)
                f1_val = val_results.get("F1", 0.0)
                auc_val = val_results.get("auc", 0.0)
                logger.info(
                    f"Val E{epoch + 1}: Acc:{acc_val:.4f} F1:{f1_val:.4f} AUC:{auc_val:.4f} "
                    f"Tracked({self.metric_key_for_early_stop}):{tracked_metric:.4f}"
                )
                real_val = val_results.get("Real", val_results.get("real", {}))
                fake_val = val_results.get("Fake", val_results.get("fake", {}))
                logger.info(
                    f"  Real:{repr(real_val)}, "
                    f"Fake:{repr(fake_val)}"
                )

                mark = recorder.add(val_results)
                if mark == "save":
                    save_path = os.path.join(self.save_param_dir, self.model_save_filename)
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Best model saved based on '{self.metric_key_for_early_stop}': {save_path}")
                elif mark == "esc":
                    logger.info(f"Early stopping triggered based on '{self.metric_key_for_early_stop}'.")
                    break
            except Exception as exc:
                logger.exception(f"Val epoch {epoch + 1} error: {exc}")
                continue

        logger.info("Training loop finished.")
        try:
            recorder.showfinal()
        except Exception as exc:
            logger.warning(f"Recorder.showfinal failed: {exc}")


        model_path = os.path.join(self.save_param_dir, self.model_save_filename)
        loaded_best = False
        if os.path.exists(model_path):
            logger.info(f"Loading best model for final test: {model_path}")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                loaded_best = True
            except Exception as exc:
                logger.error(f"Failed to load best model state_dict: {exc}. Using model state from end of training.")

        if not loaded_best:
            model_path = os.path.join(self.save_param_dir, "final_training_state_d2ian_finefake.pth")
            logger.warning(
                f"Best model not loaded. Testing with model state from end of training. "
                f"Saving this state to: {model_path}"
            )
            try:
                torch.save(self.model.state_dict(), model_path)
            except Exception as exc:
                logger.error(f"Failed to save final training state model: {exc}")

        final_results = {}
        if self.test_loader is None:
            logger.warning("Test loader not provided. Skipping final test.")
            final_results = recorder.max if hasattr(recorder, "max") else {}
        else:
            logger.info("Starting final test with the chosen model...")
            try:
                final_results = self.test(self.test_loader)
                if final_results:
                    print(f"Final Test Results: {final_results}")
                else:
                    logger.error("Final test did not return valid results.")
            except Exception as exc:
                logger.exception(f"Final test execution error: {exc}")

        return final_results, model_path

    def test(self, dataloader):

        if dataloader is None:
            logger.error("Test dataloader is None.")
            return {}

        self.model.eval()
        all_preds, all_labels, all_categories = [], [], []
        all_conflict_weights = []
        all_weight_categories = []

        with torch.no_grad():
            for step_n, batch in enumerate(tqdm.tqdm(dataloader, desc="Testing")):
                try:
                    batch_data = self._prepare_batch(batch)
                    if batch_data is None:
                        logger.warning(f"Skipping test batch {step_n} due to data loading/GPU transfer error.")
                        continue
                    if "label" not in batch_data:
                        logger.warning(f"Skipping test batch {step_n} due to missing label.")
                        continue

                    final_logits, _, _, _, _, conflict_weights, _ = self.model(**batch_data)
                    probs = torch.sigmoid(final_logits)

                    all_preds.extend(probs.detach().cpu().numpy().tolist())
                    all_labels.extend(batch_data["label"].detach().cpu().numpy().tolist())
                    all_conflict_weights.append(conflict_weights.detach().cpu())

                    if "category" in batch_data and isinstance(batch_data["category"], torch.Tensor):
                        batch_categories = batch_data["category"].detach().cpu()
                        all_categories.extend(batch_categories.numpy().tolist())
                        all_weight_categories.extend(batch_categories.tolist())
                    else:
                        logger.debug(
                            f"Test batch {step_n} missing category, will pass None to metrics if category_dict used."
                        )
                        all_categories.extend([None] * batch_data["label"].size(0))
                        all_weight_categories.extend([None] * batch_data["label"].size(0))
                except Exception as exc:
                    logger.exception(f"Test batch {step_n} error: {exc}")
                    continue

        if not all_labels:
            logger.warning("No valid test samples were processed. Metrics cannot be calculated.")
            return {}


        if self.category_dict and len(all_categories) != len(all_labels):
            logger.warning(
                f"Mismatch category/label length ({len(all_categories)} vs {len(all_labels)}). "
                "Filling category list with None."
            )
            all_categories = (all_categories + [None] * len(all_labels))[: len(all_labels)]

        try:
            if self.category_dict:
                results = calculate_metrics(all_labels, all_preds, all_categories, self.category_dict)
            else:
                results = calculate_metrics(all_labels, all_preds)
            if all_conflict_weights:
                weights = torch.cat(all_conflict_weights, dim=0)
                mean_weights = weights.mean(dim=0)
                results["LL_weight"] = float(mean_weights[0].item())
                results["GL_weight"] = float(mean_weights[1].item())
                results["GG_weight"] = float(mean_weights[2].item())
                print(
                    "Mean conflict weights: "
                    f"LL={results['LL_weight']:.4f}, "
                    f"GL={results['GL_weight']:.4f}, "
                    f"GG={results['GG_weight']:.4f}"
                )
                if self.category_dict and len(all_weight_categories) == weights.size(0):
                    categories = torch.tensor(
                        [int(category) if category is not None else -1 for category in all_weight_categories],
                        dtype=torch.long,
                    )
                    results["domain_conflict_weights"] = {}
                    for domain_name, domain_id in self.category_dict.items():
                        mask = categories == int(domain_id)
                        if not torch.any(mask):
                            continue
                        domain_mean = weights[mask].mean(dim=0)
                        domain_weights = {
                            "LL": float(domain_mean[0].item()),
                            "GL": float(domain_mean[1].item()),
                            "GG": float(domain_mean[2].item()),
                        }
                        results["domain_conflict_weights"][domain_name] = domain_weights
                        print(
                            f"Mean conflict weights [{domain_name}]: "
                            f"LL={domain_weights['LL']:.4f}, "
                            f"GL={domain_weights['GL']:.4f}, "
                            f"GG={domain_weights['GG']:.4f}"
                        )
            return results
        except Exception as exc:
            logger.exception(f"Metrics calculation error: {exc}")
            return {}
