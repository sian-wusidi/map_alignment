
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelLevelModule,
)
class TemporalPixelLevelModule(Mask2FormerPixelLevelModule):
    """
    Extends Mask2FormerPixelLevelModule to handle temporal sequences.
    Adds cross-attention fusion at each feature level.
    """
    def __init__(self, config):
        super().__init__(config)
        self.fusion_dim = 256 
        ffn_dim = 512
        
        # Cross-attention + FFN for temporal fusion (shared across feature levels)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim, 
            num_heads=8, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.fusion_dim)
        
        # FFN: two linear transformations with ReLU and dropout
        self.linear1 = nn.Linear(self.fusion_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, self.fusion_dim)
        self.norm2 = nn.LayerNorm(self.fusion_dim)

    def _fuse_time_series(self, features):
        """
        Fuse temporal features using cross-attention.
        Args:
            features: [Time, C, H, W] tensor
        Returns:
            fused: [1, C, H, W] tensor
        """
        T, C, H, W = features.shape
        
        # Use last frame as query, previous frames as key/value
        target = features[-1:]  # [1, C, H, W]
        refs = features[:-1]    # [T-1, C, H, W]
        
        # Reshape for attention: query is [H*W, 1, C]
        q = target.permute(0, 2, 3, 1).reshape(H * W, 1, C)
        
        if len(refs) > 0:
            # Reshape references: key/value is [H*W, T-1, C]
            k_v = refs.permute(2, 3, 0, 1).reshape(H * W, T - 1, C)
            
            # Cross-attention
            attn_out, _ = self.multihead_attn(query=q, key=k_v, value=k_v)
            x = self.norm1(q + attn_out)
        else:
            # Single frame case (no temporal context)
            x = self.norm1(q)

        # FFN with ReLU and dropout
        x_ffn = self.linear2(F.dropout(F.relu(self.linear1(x)), p=0.1))
        x = self.norm2(x + x_ffn)
        
        # Reshape back to [1, C, H, W]
        return x.reshape(1, H, W, C).permute(0, 3, 1, 2)

    def forward(self, pixel_values_list, pixel_mask=None):
        """
        Process batch of temporal sequences.
        Args:
            pixel_values_list: List of [Time, C, H, W] tensors
        Returns:
            tuple: Same as parent class (encoder_features, decoder_features)
        """
        batch_fused_mask = []
        batch_fused_multi = [[], [], []]
        
        # Process each sample's time series
        for time_series_tensor in pixel_values_list:
            # Extract features via backbone encoder
            backbone_out = self.encoder(time_series_tensor, output_hidden_states=True)
            
            # Decode to multi-scale features
            decoder_out = self.decoder(backbone_out.feature_maps)
            
            # decoder_out structure: decoder_out[0] is multi-scale features (tuple of 3)
            # decoder_out[1] is the mask features
            multiscale_features = decoder_out[0]  # Tuple of 3 scales
            mask_features = decoder_out[1]  # Main mask features
            
            # Fuse temporal information at each feature level
            batch_fused_mask.append(self._fuse_time_series(mask_features))
            batch_fused_multi[0].append(self._fuse_time_series(multiscale_features[0]))
            batch_fused_multi[1].append(self._fuse_time_series(multiscale_features[1]))
            batch_fused_multi[2].append(self._fuse_time_series(multiscale_features[2]))
        
        # Concatenate all samples in batch
        final_mask_features = torch.cat(batch_fused_mask, dim=0)
        final_multiscale = (
            torch.cat(batch_fused_multi[0], dim=0),
            torch.cat(batch_fused_multi[1], dim=0),
            torch.cat(batch_fused_multi[2], dim=0)
        )
        
        # Return in the same format as parent class
        return final_mask_features, final_multiscale


class TemporalMask2Former(Mask2FormerForUniversalSegmentation):
    """
    Modified Mask2FormerForUniversalSegmentation that supports temporal sequences.
    Only changes: pixel_level_module handles list input with temporal fusion.
    """
    def __init__(self, config):
        super().__init__(config)
        # Replace pixel module with temporal version
        self.model.pixel_level_module = TemporalPixelLevelModule(config)
        self.post_init()

    def forward(
        self,
        pixel_values,  # List of [Time, C, H, W] tensors (temporal input)
        mask_labels=None,
        class_labels=None,
        pixel_mask=None,
        output_hidden_states=None,
        output_auxiliary_logits=None,
        output_attentions=None,
        return_dict=None,
    ):
        """
        Forward pass with temporal input support.
        Rest follows original Mask2FormerForUniversalSegmentation exactly.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get batch size from list length
        batch_size = len(pixel_values)
        
        # Run temporal pixel module (handles fusion internally)
        mask_features, multi_scale_features = self.model.pixel_level_module(pixel_values, pixel_mask)
        
        # Create pixel_mask if not provided (needed for transformer)
        if pixel_mask is None:
            _, _, height, width = mask_features.shape
            pixel_mask = torch.ones((batch_size, height, width), device=mask_features.device, dtype=torch.long)
        
        # Run transformer module (unchanged from original)
        transformer_out = self.model.transformer_module(
            multi_scale_features=multi_scale_features,
            mask_features=mask_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
        )
        
        # Package outputs in expected format
        from transformers.models.mask2former.modeling_mask2former import Mask2FormerModelOutput
        outputs = Mask2FormerModelOutput(
            encoder_last_hidden_state=None,
            pixel_decoder_last_hidden_state=mask_features,
            transformer_decoder_last_hidden_state=transformer_out.last_hidden_state,
            transformer_decoder_intermediate_states=transformer_out.intermediate_hidden_states,
            masks_queries_logits=transformer_out.masks_queries_logits,
            attentions=transformer_out.attentions if hasattr(transformer_out, "attentions") else None,
        )

        # === Rest is identical to original Mask2FormerForUniversalSegmentation ===
        
        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states if hasattr(outputs, 'encoder_hidden_states') else None
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states if hasattr(outputs, 'pixel_decoder_hidden_states') else None
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states if hasattr(outputs, 'transformer_decoder_hidden_states') else None

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_logits = None

        from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values() if v is not None)
            if loss is not None:
                output = (loss,) + output
        return output