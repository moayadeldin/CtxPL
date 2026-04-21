import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import torch.nn.init as torch_init

from edl_loss import EvidenceLoss
from edl_loss import relu_evidence, exp_evidence, softplus_evidence

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 1, (1,)),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat


class CTXPL(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio
        self.activate_HNCPs = args['opt'].activate_HNCPs if hasattr(args['opt'], 'activate_HNCPs') else False
        

        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, (1,), padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(embed_dim, n_class + 1, (1,))
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_prototypes = {}
        self.class_background_prototypes = {}
        self.register_buffer('prototype_counts', torch.zeros(n_class))
        
        # Uncertainty threshold for prototype selection
        self.uncertainty_threshold = args['opt'].uncertainty_threshold if hasattr(args['opt'], 'uncertainty_threshold') else 0.7
        
        # Prototype momentum for update
        self.prototype_momentum = args['opt'].prototype_momentum if hasattr(args['opt'], 'prototype_momentum') else 0.9
        
        # Prototype loss weight
        self.prototype_loss_weight = args['opt'].prototype_loss_weight if hasattr(args['opt'], 'prototype_loss_weight') else 1
        
        ### MULTIPLE PROTOTYPES ###
        self.max_prototypes_per_class = args['opt'].max_prototypes_per_class if hasattr(args['opt'], 'max_prototypes_per_class') else 6

        
        print(self.max_prototypes_per_class)

        self.max_background_prototypes_per_class = args['opt'].max_prototypes_per_class if hasattr(args['opt'], 'max_background_prototypes_per_class') else 8

        ### ADDING CONTEXT PART ### 
        
        self.ctx_topk_ratio = getattr(args['opt'], 'ctx_topk_ratio', 0.10)   # top-k ≈ 10% of T
        self.ctx_tau        = getattr(args['opt'], 'ctx_tau', 0.05)          # softness for top-k
        self.ctx_a          = getattr(args['opt'], 'ctx_a', 0.9)             # foreground weight
        self.ctx_b          = getattr(args['opt'], 'ctx_b', 0.1)             # context weight

        self.apply(weights_init)
        
    def _contextual_video_repr(
        self,
        video_feat_TD,
        cas_TC,
        cls_idx,
        u_snip_T
    ):
        
        """
        Build X = row_norm(W[:,None] ⊙ X_v), then average over time to (D,).
        W_t = a * m_fg(t) + b * m_ctx(t), where
        m_fg = soft-topk(CAS), m_ctx = (1 - m_fg) * (1 - p_norm) * (1 - u_t).
        """
        
        T,D = video_feat_TD.shape
        p = cas_TC[:, cls_idx]
        p = p.sigmoid() if p.min() < 0 or p.max() > 1 else p  # handle logits vs probs
        
        
        k = max(1, int(self.ctx_topk_ratio * T))
        
        # kth largest value 
        theta, _ = torch.kthvalue(p, T - k + 1)
        m_fg = torch.sigmoid((p - theta) / self.ctx_tau)

        p_min, p_max = p.min(), p.max()
        denom = (p_max - p_min + 1e-6)
        p_norm = (p - p_min) / denom

        # per-snippet uncertainty
        if (u_snip_T is None) or (not self.use_snip_unc):
            u_snip = torch.zeros_like(p)
        else:
            u_snip = u_snip_T.clamp_(0, 1)

        m_ctx = (1.0 - m_fg) * (1.0 - p_norm) * (1.0 - u_snip)  # (T,)

        W = self.ctx_a * m_fg + self.ctx_b * m_ctx              # (T,)
        
        W = W + 1e-6
        W = W / W.sum()

        # weighted features -> (T, D), then average to (D,)
        X_TD = video_feat_TD * W.unsqueeze(1)
        X_D  = X_TD.mean(dim=0)                                 # (D,)
        return X_D

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        outputs = {'feat': nfeat.transpose(-1, -2),   # (B, T, D=2048)
                   'cas': x_cls.transpose(-1, -2),    # (B, T, C+1)
                   'attn': x_atn.transpose(-1, -2),
                   'v_atn': v_atn.transpose(-1, -2),
                   'f_atn': f_atn.transpose(-1, -2),
                   }
        return outputs

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)

        outputs = {'feat': nfeat.transpose(-1, -2),
                   'cas': x_cls.transpose(-1, -2),
                   'attn': x_atn.transpose(-1, -2),
                   'v_atn': v_atn.transpose(-1, -2),
                   'f_atn': f_atn.transpose(-1, -2),
                   }

        return outputs

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min
    
    def _compute_similarity_threshold(self, feats, labels):
        """
        Compute dynamic similarity threshold for prototype similarity.
        """
        distances = []
        
        video_reprs = self._get_video_representation(feats)  # (B, D)
        norms = torch.norm(video_reprs, dim=1)
        
        

        for i in range(feats.shape[0]):  # B
            active_classes = torch.where(labels[i] > 0)[0]
            video_feat = self._get_video_representation(feats[i].unsqueeze(0))  # (1, D)
            for cls_idx_tensor in active_classes:
                cls_idx = int(cls_idx_tensor)
                if cls_idx in self.class_prototypes and self.class_prototypes[cls_idx]["prototypes"]:
                    protos = torch.stack(self.class_prototypes[cls_idx]["prototypes"])  # (K, D)
                    dists = torch.norm(protos - video_feat, dim=1)  # (K,)
                    min_dist = torch.min(dists).item()
                    distances.append(min_dist)

        if len(distances) == 0:
            return 1.0

        mean = np.mean(distances)
        return mean

            

    def _get_video_representation(self, feat):
        """Average pool the feature tensor to get a video-level representation"""
        return torch.mean(feat, dim=1)  # B x C
    
    def _compute_video_uncertainty(self, element_logits_supp, n_class):
        """Compute video-level uncertainty from evidence"""
        # Average pooling over snippets
        video_logits = torch.mean(element_logits_supp, dim=1)
        
        # Convert to evidence
        evidence = exp_evidence(video_logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        
        # Uncertainty calculation (higher value = more uncertain)
        uncertainty = n_class / S
        
        return uncertainty
    
    
    
    def _update_prototypes(self, feat, labels, uncertainty, n_class, cas=None, snippet_uncertainty=None):
        batch_size = feat.shape[0]
        video_repr = self._get_video_representation(feat)  # (B, D)
        prototype_similarity_threshold = self._compute_similarity_threshold(feat,labels)


        for i in range(batch_size):
            
            active_classes = torch.where(labels[i, :n_class] > 0)[0]
            if len(active_classes) == 0:
                continue
            for cls_idx_tensor in active_classes:
                cls_idx = int(cls_idx_tensor)
            
                if uncertainty[i] > self.uncertainty_threshold:
                    
                    # => Activate HNCPs
                    if self.activate_HNCPs:
                     
                        entry = self.class_background_prototypes.setdefault(cls_idx, {
                            "prototypes":[],
                            "uncertainties":[]
                        })
                        
                        if len(entry["prototypes"]) < self.max_background_prototypes_per_class:
                            entry["prototypes"].append(video_repr[i].detach())
                            entry["uncertainties"].append(uncertainty[i])
                            
                        else:
                            unc_tensor = torch.tensor(entry["uncertainties"], device=video_repr.device)
                            least_uncertain_idx = torch.argmin(unc_tensor).item()
                            if uncertainty[i] < entry["uncertainties"][least_uncertain_idx]:
                                    entry["prototypes"][least_uncertain_idx] = video_repr[i].detach()
                                    entry["uncertainties"][least_uncertain_idx] = uncertainty[i]
                                    
                    else:
                        continue
                else:
                    
                
                    # prepare per-sample tensors
                    feat_TD = feat[i]                       # (T, D)
                    cas_TC1 = cas[i] if cas is not None else None  # (T, C+1)
                    # slice out background column if present
                    cas_TC = cas_TC1[:, :n_class] if (cas_TC1 is not None and cas_TC1.shape[-1] == n_class + 1) else cas_TC1
                    # print(cas_TC)
                    u_snip_T = None if snippet_uncertainty is None else snippet_uncertainty[i]  # (T,)
                    entry = self.class_prototypes.setdefault(cls_idx, {
                        "prototypes": [],
                        "uncertainties": [],
                        "counts": []
                    })
                    
                    def _ctx_vec():
                        if cas_TC is None:
                            return video_repr[i].detach()
                        
                        return self._contextual_video_repr(
                            video_feat_TD=feat_TD,
                            cas_TC=cas_TC,
                            cls_idx=cls_idx,
                            u_snip_T=u_snip_T
                        ).detach()
                    
                    if len(entry["prototypes"]) == 0:
                        # No prototypes yet, just add this one
                        entry["prototypes"].append(_ctx_vec())
                        entry["uncertainties"].append(uncertainty[i])
                        entry["counts"].append(1)
                        continue

                    proto_tensor = torch.stack(entry["prototypes"])  # (K, D)
                    dists = torch.norm(proto_tensor - video_repr[i].unsqueeze(0), dim=1)  # (K,)
                    min_dist, min_idx = torch.min(dists, dim=0)

                    if min_dist < prototype_similarity_threshold:
                        # Update the closest prototype using momentum
                        proto = proto_tensor[min_idx]
                        updated_proto = proto * self.prototype_momentum + video_repr[i] * (1 - self.prototype_momentum)
                        entry["prototypes"][min_idx] = updated_proto
                        entry["uncertainties"][min_idx] = min(entry["uncertainties"][min_idx], uncertainty[i])
                        entry["counts"][min_idx] += 1
                    else:
                        
                        ctx_vec = _ctx_vec()
                        if len(entry["prototypes"]) < self.max_prototypes_per_class:
                            entry["prototypes"].append(ctx_vec)
                            entry["uncertainties"].append(uncertainty[i])
                            entry["counts"].append(1)
                        else:
                            # Replace highest-uncertainty prototype if current one is better
                            unc_tensor = torch.tensor(entry["uncertainties"], device=video_repr.device)
                            worst_idx = torch.argmax(unc_tensor).item()
                            if uncertainty[i] < entry["uncertainties"][worst_idx]:
                                entry["prototypes"][worst_idx] = video_repr[i].detach()
                                entry["uncertainties"][worst_idx] = uncertainty[i]
                                entry["counts"][worst_idx] = 1
    
    #=> PAR Loss

    def prototype_attraction_repulsion_loss(self, feat, labels, class_prototypes, class_background_prototypes, uncertainty, n_class):
        
        B = feat.shape[0]
        
        temperature = 0.3
        total_loss = 0.0
        count = 0
        
        video_repr = self._get_video_representation(feat)
        video_repr = F.normalize(video_repr, dim=1)
        
        for i in range(B):
            
            # if uncertainty[i] > self.uncertainty_threshold:
            #     continue
            
            active_classes = torch.where(labels[i, :n_class] > 0)[0]
            if len(active_classes) == 0:
                continue
            
            anchor = video_repr[i].unsqueeze(0)
            
            for cls_idx_tensor in active_classes:
                
                cls_idx = int(cls_idx_tensor)
                
                fg_entry = class_prototypes.get(cls_idx)
                bg_entry = class_background_prototypes.get(cls_idx)
                
                if not fg_entry or not fg_entry["prototypes"] or not bg_entry or not bg_entry["prototypes"]:
                    continue
                
                fg_protos = torch.stack(fg_entry["prototypes"])
                bg_protos = torch.stack(bg_entry["prototypes"])
                
                fg_protos = F.normalize(fg_protos, dim=1)
                bg_protos = F.normalize(bg_protos, dim=1)
                
                sim_fg = F.cosine_similarity(anchor, fg_protos) / temperature
                sim_bg = F.cosine_similarity(anchor, bg_protos) / temperature
                
                s_pos = torch.sum(torch.exp(sim_fg))
                s_neg = torch.sum(torch.exp(sim_bg))
                
                loss = -torch.log(s_pos / (s_pos + s_neg))
                total_loss += loss
                count += 1
                

        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=video_repr.device)

    def prototype_alignment_loss(self, feat, labels, uncertainty, n_class):
        B, D = feat.shape[0], feat.shape[2]
        video_repr = self._get_video_representation(feat)  # (B, D)
        prototype_loss = []
        
        for i in range(B):
            if uncertainty[i] > self.uncertainty_threshold:
                continue

            active_classes = torch.where(labels[i, :n_class] > 0)[0]
            if len(active_classes) == 0:
                continue
            
            cls_prototype_losses = []

            for cls_idx in active_classes:
                cls_idx = int(cls_idx)
                entry = self.class_prototypes.get(cls_idx, {"prototypes": []})
                if not entry["prototypes"]:
                    continue

                protos = torch.stack(entry["prototypes"])  # (K, D)
                
                if protos.shape[1] != video_repr[i].shape[0]:
                    expanded_vid = video_repr[i].expand(protos.shape[0], -1)
                else:
                    expanded_vid = video_repr[i].expand_as(protos)
                
                try:
                    dists = F.mse_loss(expanded_vid, protos, reduction='none').mean(dim=1)  # (K,)
                    min_dist = torch.min(dists)
                    cls_prototype_losses.append(min_dist)
                except Exception as e:
                    print(f"Error computing distances: {e}")
                    
            if cls_prototype_losses:
                sample_loss = torch.stack(cls_prototype_losses).mean()
                prototype_loss.append(sample_loss)

        if not prototype_loss:
            return torch.tensor(0., device=feat.device)
        
        final_loss = torch.stack(prototype_loss).mean()
        return final_loss


    def criterion(self, outputs, labels, **args):
        
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())

        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)

        edl_loss = self.edl_loss(element_logits_supp,
                                 element_atn,
                                 labels,
                                 rat=args['opt'].rat_atn,
                                 n_class=args['opt'].num_class,
                                 epoch=args['itr'],
                                 total_epoch=args['opt'].max_iter,
                                 )

        snippet_uct, uct_guide_loss = self.uct_guide_loss(element_logits,
                                             element_logits_supp,
                                             element_atn,
                                             v_atn,
                                             f_atn,
                                             n_class=args['opt'].num_class,
                                             epoch=args['itr'],
                                             total_epoch=args['opt'].max_iter,
                                             amplitude=args['opt'].amplitude,
                                             )

        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k)

        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()
        
        # Compute video uncertainty for prototype handling
        n_class = args['opt'].num_class
        video_uncertainty = self._compute_video_uncertainty(element_logits_supp, n_class)
        
        # Update prototypes
        with torch.no_grad():
            self._update_prototypes(feat, labels,video_uncertainty, n_class, cas=outputs['cas'], snippet_uncertainty=snippet_uct)
        
        # Compute prototype loss
        prototype_loss = self.prototype_alignment_loss(feat, labels, video_uncertainty, n_class)
        
        if self.activate_HNCPs:
        
            prototype_attraction_repulsion_loss = self.prototype_attraction_repulsion_loss(feat, labels, self.class_prototypes, self.class_background_prototypes, video_uncertainty, n_class)


            total_loss = (
                        args['opt'].alpha_edl * edl_loss +
                        args['opt'].alpha_uct_guide * uct_guide_loss +
                        loss_mil_orig.mean() + loss_mil_supp.mean() +
                        args['opt'].alpha3 * loss_3_supp_Contrastive +
                        args['opt'].alpha4 * mutual_loss +
                        args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                        args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3 +
                        self.prototype_loss_weight * prototype_loss +
                        prototype_attraction_repulsion_loss)

            loss_dict = {
                'edl_loss': args['opt'].alpha_edl * edl_loss,
                'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
                'loss_mil_orig': loss_mil_orig.mean(),
                'loss_mil_supp': loss_mil_supp.mean(),
                'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
                'mutual_loss': args['opt'].alpha4 * mutual_loss,
                'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
                'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
                'prototype_loss': self.prototype_loss_weight * prototype_loss,
                'prototype_attraction_repulsion_loss': prototype_attraction_repulsion_loss,
                'total_loss': total_loss,
            }
        else:
        
            total_loss = (
                        args['opt'].alpha_edl * edl_loss +
                        args['opt'].alpha_uct_guide * uct_guide_loss +
                        loss_mil_orig.mean() + loss_mil_supp.mean() +
                        args['opt'].alpha3 * loss_3_supp_Contrastive +
                        args['opt'].alpha4 * mutual_loss +
                        args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                        args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3 +
                        self.prototype_loss_weight * prototype_loss)

            loss_dict = {
                'edl_loss': args['opt'].alpha_edl * edl_loss,
                'uct_guide_loss': args['opt'].alpha_uct_guide * uct_guide_loss,
                'loss_mil_orig': loss_mil_orig.mean(),
                'loss_mil_supp': loss_mil_supp.mean(),
                'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
                'mutual_loss': args['opt'].alpha4 * mutual_loss,
                'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
                'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
                'prototype_loss': self.prototype_loss_weight * prototype_loss,
                'total_loss': total_loss,
            }
        

        return total_loss, loss_dict

    def uct_guide_loss(self,
                       element_logits,
                       element_logits_supp,
                       element_atn,
                       v_atn,
                       f_atn,
                       n_class,
                       epoch,
                       total_epoch,
                       amplitude):

        evidence = exp_evidence(element_logits_supp)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1)
        snippet_uct = n_class / S

        total_snippet_num = element_logits.shape[1]
        curve = self.course_function(epoch, total_epoch, total_snippet_num, amplitude)

        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().squeeze()

        total_loss_guide = (loss_guide + v_loss_guide + f_loss_guide) / 3

        _, uct_indices = torch.sort(snippet_uct, dim=1)
        sorted_curve = torch.gather(curve.repeat(10, 1), 1, uct_indices)

        uct_guide_loss = torch.mul(sorted_curve, total_loss_guide).mean()

        return snippet_uct, uct_guide_loss

    def edl_loss(self,
                 element_logits_supp,
                 element_atn,
                 labels,
                 rat,
                 n_class,
                 epoch=0,
                 total_epoch=5000,
                 ):

        k = max(1, int(element_logits_supp.shape[-2] // rat))

        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )
        atn_idx_expand = atn_idx.expand([-1, -1, n_class + 1])
        topk_element_logits = torch.gather(element_logits_supp, 1, atn_idx_expand)[:, :, :-1]
        video_logits = topk_element_logits.mean(dim=1)

        edl_loss = EvidenceLoss(
            num_classes=n_class,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            disentangle=False,
            annealing_method='exp')

        edl_results = edl_loss(
            output=video_logits,
            target=labels,
            epoch=epoch,
            total_epoch=total_epoch
        )

        edl_loss = edl_results['loss_cls'].mean()

        return edl_loss

    def course_function(self, epoch, total_epoch, total_snippet_num, amplitude):

        idx = torch.arange(total_snippet_num)
        theta = 2 * (idx + 0.5) / total_snippet_num - 1
        delta = - 2 * epoch / total_epoch + 1
        curve = amplitude * torch.tanh(theta * delta) + 1

        return curve

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 rat=8):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(topk_val, dim=-2)

        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = - (labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1)

        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn