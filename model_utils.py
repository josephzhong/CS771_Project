import os
import torch
from peft import get_peft_model, LoraConfig
from torch import nn
from typing import Optional, Unpack, List, Union, Any
from transformers import GenerationMixin, AutoModelForCausalLM, Qwen3VLModel, Cache, \
    Qwen3VLPreTrainedModel, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast
from transformers.utils import TransformersKwargs


class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


def load_base_model(model_name, save_path=None, attn_implementation=None, dtype=None, max_length=None):
    if save_path is None:
        save_path = model_name
    if dtype is None:
        dtype = torch.float32
    if model_name.find("Qwen3-VL") > -1:
        base_cls = Qwen3VLForConditionalGeneration
    else:
        # TODO: needs some special processing to save GPU RAM
        base_cls = AutoModelForCausalLM
    return base_cls.from_pretrained(save_path, trust_remote_code=True,
                                    attn_implementation=attn_implementation, dtype=dtype)


class R2T(nn.Module):
    """Trainable Reward hacking Retrieval Tokens: shape (R, H)."""

    def __init__(self, num_tokens: int, hidden_size: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Parameter(torch.randn(num_tokens, hidden_size) * 0.02, requires_grad=True)

    def init_weights(self, tok_embedding_weight):
        # print(f"tok_embedding_weight size {tok_embedding_weight.size()}")
        # print(f"finite value count {torch.isfinite(tok_embedding_weight).float().sum().item()}")
        mu = tok_embedding_weight.mean()  # (1, H)
        print(f"R2T init mu = {mu.item()}")
        sigma = tok_embedding_weight.std() + 1e-6
        print(f"R2T init sigma = {sigma.item()}")
        with torch.no_grad():
            self.embedding.copy_(mu + torch.randn_like(self.embedding) * sigma)

    def forward(self, bsz: int, dtype):
        return self.embedding.to(dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)


class MMR2TMetaEmbed(nn.Module):
    """
    - Shared backbone with LoRA
    - Separate meta-token blocks for query (Rq) and candidate (Rc)
    - Appends meta tokens to end; extracts last-layer hidden states at those positions
    """

    def __init__(self, base_model, tokenizer, groups_q: List[int], groups_c: List[int], temperature: float,
                 lora_cfg: LoraConfig | None, soft_weight: float,
                 load=False, backbone=None, r2t_q=None, r2t_c=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.groups_q = groups_q
        self.groups_c = groups_c
        self.temperature = temperature
        self.Rq = max(groups_q)
        self.Rc = max(groups_c)
        # for current left padding design
        assert self.Rq == self.Rc
        if load:
            self.soft_weight = soft_weight
            self.backbone = backbone
            self.hidden_size = self.backbone.language_model.config.hidden_size
            self.r2t_q = r2t_q
            self.r2t_c = r2t_c
        else:
            self.soft_weight = soft_weight
            self.backbone = get_peft_model(base_model, lora_cfg)
            self.hidden_size = self.backbone.language_model.config.hidden_size
            tok_emb = self.backbone.get_input_embeddings()
            weight = tok_emb.weight
            self.r2t_q = R2T(self.Rq, self.hidden_size).to(device=weight.device)
            self.r2t_c = R2T(self.Rc, self.hidden_size).to(device=weight.device)
            self.r2t_q.init_weights(weight)
            self.r2t_c.init_weights(weight)

    def save_model(self, save_dir, groups_q, groups_c, tokenizer=None):
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir, base_model=self.backbone.base_model.config._name_or_path)
        torch.save({"groups_q": groups_q, "state_dict": self.r2t_q.state_dict()}, os.path.join(save_dir, "r2t_q.pt"))
        torch.save({"groups_c": groups_c, "state_dict": self.r2t_c.state_dict()}, os.path.join(save_dir, "r2t_c.pt"))
        if tokenizer:
            tokenizer.save_pretrained(save_dir)
        print(f"Saved LoRA adapter + meta tokens to {save_dir}")

    @staticmethod
    def load_model(model_name, save_path, tokenizer, temperature, attn_implementation, soft_weight, device, dtype):
        backbone = load_base_model(model_name, save_path, attn_implementation=attn_implementation, dtype=dtype)
        groups_q_dict = torch.load(os.path.join(save_path, "r2t_q.pt"), map_location=device)
        groups_q = groups_q_dict["groups_q"]
        print(f"info: groups_q: {groups_q}")
        r2t_q = R2T(max(groups_q), backbone.language_model.config.hidden_size)
        r2t_q.load_state_dict(groups_q_dict["state_dict"])
        groups_c_dict = torch.load(os.path.join(save_path, "r2t_c.pt"), map_location=device)
        groups_c = groups_c_dict["groups_c"]
        print(f"info: groups_c: {groups_c}")
        r2t_c = R2T(max(groups_c), backbone.language_model.config.hidden_size)
        r2t_c.load_state_dict(groups_c_dict["state_dict"])
        model = MMR2TMetaEmbed(base_model=None, tokenizer=tokenizer, groups_q=groups_q, groups_c=groups_c, temperature=temperature,
                               lora_cfg=None, soft_weight=soft_weight, load=True, backbone=backbone, r2t_q=r2t_q, r2t_c=r2t_c)
        model.to(device)
        return model, tokenizer

    def _encode_with_meta(self, input_ids, r2t_starts, attention_mask, r2t_module):
        """
        Append meta tokens to embeddings, extend mask with ones,
        forward with inputs_embeds, return last hidden states of meta span.
        """
        bsz = input_ids.size(0)
        tok_emb = self.backbone.get_input_embeddings()(input_ids)  # (B, L, H)
        hidden_size = tok_emb.size(2)
        r2t_emb = r2t_module(bsz, dtype=tok_emb.dtype)  # (B, R, H)
        pos = torch.arange(self.Rq, device=r2t_starts.device).unsqueeze(0)  # [1, R]
        cols = r2t_starts.unsqueeze(1) + pos  # [B, R] absolute positions
        cols = cols.unsqueeze(2).expand(-1, -1, hidden_size)
        tok_emb = tok_emb.scatter(dim=1, index=cols, src=r2t_emb)

        out = self.backbone(
            inputs_embeds=tok_emb,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        last = out.last_hidden_state  # (B, L, H)
        r2t_span = torch.gather(last, dim=1, index=cols)  # (B, R, H)
        r2t_span = nn.functional.normalize(r2t_span, dim=-1)

        return r2t_span  # multi-vector embedding

    def encode_query(self, input_ids, r2t_starts, attention_mask):
        return self._encode_with_meta(input_ids, r2t_starts, attention_mask, self.r2t_q)

    def encode_candidate(self, input_ids, r2t_starts, attention_mask):
        return self._encode_with_meta(input_ids, r2t_starts, attention_mask, self.r2t_c)

    def forward(self, batch, neg=True, rq=None, rc=None, target_step_size=2):
        if neg:
            if self.training:
                r2t_span_q, r2t_span_c, r2t_span_c_neg = self.encode_query_candidate_and_neg(batch, target_step_size=target_step_size)
                return self.compute_loss_group(r2t_span_q, r2t_span_c, r2t_span_c_neg)
            else:
                r2t_span_q, r2t_span_c, r2t_span_c_neg = self.encode_query_candidate_and_neg(batch, target_step_size=target_step_size)
                s = (self.inference_score(r2t_span_q, r2t_span_c, rq, rc) / rq + 1) / 2.0 # normalize score for testing
                s_neg = (self.inference_score_neg(r2t_span_q, r2t_span_c_neg, rq, rc) / rq + 1) / 2.0 # normalize score for testing
                return s, s_neg, r2t_span_q, r2t_span_c, r2t_span_c_neg
        else:
            assert not self.training
            r2t_span_q, r2t_span_c = self.encode_query_candidate_and_neg(batch, neg=False, target_step_size=target_step_size)
            s = (self.inference_score(r2t_span_q, r2t_span_c, rq, rc) / rq + 1) / 2.0 # normalize score for testing
            return s, None, r2t_span_q, r2t_span_c


    def encode_query_candidate_and_neg(self, batch, neg=True, target_step_size=2):
        r2t_span_q_list, r2t_span_c_list, r2t_span_c_neg_list = [], [], []
        batch_size = batch["q_input_ids"].shape[0]
        pixel_values_start_index = 0
        for start_index in range(0, batch_size, target_step_size):
            step_size = min(batch_size - start_index, target_step_size)
            q_ids = batch["q_input_ids"][start_index:start_index + step_size]
            c_ids = batch["c_input_ids"][start_index:start_index + step_size]
            image_grid_thw = batch["image_grid_thw"][start_index:start_index + step_size]
            pixel_values_step = (image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum()
            pixel_values = batch["pixel_values"][pixel_values_start_index:pixel_values_start_index + pixel_values_step]
            pixel_values_start_index += pixel_values_step

            q_attention_mask = batch["attention_mask"][start_index:start_index + step_size]
            c_attention_mask = batch["attention_mask"][start_index + batch_size: start_index + batch_size + step_size]
            if neg:
                c_neg_ids = batch["c_neg_input_ids"][start_index:start_index + step_size]
                c_neg_attention_mask = batch["attention_mask"][
                    start_index + 2 * batch_size: start_index + 2 * batch_size + step_size]
                attention_mask = torch.cat([q_attention_mask, c_attention_mask, c_neg_attention_mask], dim=0)
            else:
                attention_mask = torch.cat([q_attention_mask, c_attention_mask], dim=0)

            bsz = q_ids.size(0)
            tok_emb_q = self.backbone.model.get_input_embeddings()(q_ids)  # (B, L, H)
            r2t_q_emb = self.r2t_q(bsz, dtype=tok_emb_q.dtype)  # (B, R, H)
            tok_emb_q = torch.cat([tok_emb_q, r2t_q_emb], dim=1)

            tok_emb_c = self.backbone.model.get_input_embeddings()(c_ids)  # (B, L, H)
            r2t_c_emb = self.r2t_c(bsz, dtype=tok_emb_c.dtype)
            tok_emb_c = torch.cat([tok_emb_c, r2t_c_emb], dim=1)

            if neg:
                tok_emb_c_neg = self.backbone.model.get_input_embeddings()(c_neg_ids)  # (B, L, H)
                tok_emb_c_neg = torch.cat([tok_emb_c_neg, r2t_c_emb], dim=1)
                inputs_embeds = torch.cat([tok_emb_q, tok_emb_c, tok_emb_c_neg], dim=0)  # (3*B, L, H)
                r2t_mask = torch.ones((3 * bsz, self.r2t_q.num_tokens), dtype=r2t_q_emb.dtype, device=r2t_q_emb.device)
            else:
                inputs_embeds = torch.cat([tok_emb_q, tok_emb_c], dim=0)  # (2*B, L, H)
                attention_mask = attention_mask[:2 * bsz, :]
                r2t_mask = torch.ones((2 * bsz, self.r2t_q.num_tokens), dtype=r2t_q_emb.dtype, device=r2t_q_emb.device)
            attention_mask = torch.cat([attention_mask, r2t_mask], dim=1)
            last = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=False,
                use_cache=False,
            ).last_hidden_state # (3 * B, L, H) or (2 * B, L, H)
            r2t_span_q = last[:bsz, -1 * self.r2t_q.num_tokens:, :]
            r2t_span_c = last[bsz: 2*bsz, -1 * self.r2t_c.num_tokens:, :]
            r2t_span_q = nn.functional.normalize(r2t_span_q.float(), dim=-1).to(dtype=last.dtype)
            r2t_span_c = nn.functional.normalize(r2t_span_c.float(), dim=-1).to(dtype=last.dtype)

            r2t_span_q_list.append(r2t_span_q)
            r2t_span_c_list.append(r2t_span_c)
            if neg:
                r2t_span_c_neg = last[2 * bsz: , -1 * self.r2t_c.num_tokens:, :]  # (B, Rc, H)  # (B, Rc, H)
                r2t_span_c_neg = nn.functional.normalize(r2t_span_c_neg.float(), dim=-1).to(dtype=last.dtype)
                r2t_span_c_neg_list.append(r2t_span_c_neg)
            del tok_emb_c
            del tok_emb_q
            del r2t_q_emb
            del r2t_c_emb
            if neg:
                del tok_emb_c_neg
            del last
            del inputs_embeds
            del attention_mask
            torch.cuda.empty_cache()

        r2t_span_q = torch.cat(r2t_span_q_list, dim=0)
        r2t_span_c = torch.cat(r2t_span_c_list, dim=0)
        if neg:
            r2t_span_c_neg = torch.cat(r2t_span_c_neg_list, dim=0)
            return r2t_span_q, r2t_span_c, r2t_span_c_neg  # multi-vector embedding
        else:
            return r2t_span_q, r2t_span_c

    def matryoshka_groups(self, R: int, groups: List[int]) -> List[int]:
        uniq = sorted(list([g for g in groups if 1 <= g <= R]))
        return uniq

    def score_groups(self, Eq: torch.Tensor, Ec: torch.Tensor, Ec_neg: torch.Tensor, rq, rc) -> torch.Tensor:
        Eq_g = Eq[:, :rq, :]  # (B, rq, D)
        Ec_g = Ec[:, :rc, :]  # (B, rc, D)
        Ec_neg_g = Ec_neg[:, :rc, :]
        Ec_g_all = torch.cat([Ec_g, Ec_neg_g], dim=0)
        s = torch.sum(torch.max(torch.tensordot(Eq_g, Ec_g_all, dims=[[2], [2]]),  # (B, rq, 2 * B, rc)
                                dim=3)[0],  # (B, rq, 2 * B)
                      dim=1)  # (B, 2 * B)
        return s

    def info_nce_matryoshka(self, Eq: torch.Tensor, Ec: torch.Tensor, Ec_neg) -> torch.Tensor:
        B, Rq, D = Eq.shape
        _, Rc, _ = Ec.shape

        group_length = len(self.groups_q)

        loss_groups = torch.zeros((group_length,), device=Eq.device)

        for group_index in range(group_length):
            rq = self.groups_q[group_index]
            rc = self.groups_c[group_index]
            s_rq_rc = self.score_groups(Eq, Ec, Ec_neg, rq, rc) / self.temperature
            denominator_index = torch.cat([
                torch.arange(start=0, end=B, dtype=torch.long, device=Eq.device).unsqueeze(1),
                torch.arange(start=B, end=2 * B, dtype=torch.long, device=Eq.device).unsqueeze(1),
            ], dim=1)
            src_value = torch.ones_like(denominator_index, dtype=s_rq_rc.dtype)
            # s_rq_rc = torch.gather(s_rq_rc, dim=1, index=denominator_index)
            denominator_weight = torch.log(torch.ones_like(s_rq_rc) * self.soft_weight)
            denominator_weight = denominator_weight.scatter(dim=1, index=denominator_index, src=src_value)
            log_probs = (s_rq_rc * denominator_weight).log_softmax(dim=1)  # [B ,2*B]
            index = torch.tensor([[i] for i in range(B)], device=Eq.device)
            loss_groups[group_index] = -1 * torch.gather(log_probs, 1, index).mean()
            # loss_groups[group_index] = -1 * log_probs[:, 0].mean()

        return loss_groups

    def compute_loss_group(self, Eq, Ec, Ec_neg):
        Eq = Eq.float()
        Ec = Ec.float()
        Ec_neg = Ec_neg.float()
        loss_group = self.info_nce_matryoshka(Eq, Ec, Ec_neg)
        # loss_group = self.contrastive_matryoshka(Eq, Ec, Ec_neg)
        return loss_group

    def score_func(self, Eq_g: torch.Tensor, Ec_g: torch.Tensor) -> torch.Tensor:
        Eq_g = Eq_g.float()
        Ec_g = Ec_g.float()
        return torch.sum(torch.max(torch.matmul(Eq_g, torch.transpose(Ec_g, 1, 2)),  # (B, rq, rc)
                                   dim=2)[0],  # (B, rq)
                         dim=1)  # (B,)

    def inference_score(self, Eq: torch.Tensor, Ec: torch.Tensor, group_q_length: int, group_c_length: int) -> torch.Tensor:
        Eq_g = Eq[:, :group_q_length, :]  # (B, rq, D)
        Ec_g = Ec[:, :group_c_length, :]  # (B, rc, D)

        return self.score_func(Eq_g, Ec_g)

    def inference_score_neg(self, Eq: torch.Tensor, Ec: torch.Tensor, group_q_length: int, group_c_length: int):
        Eq_g = Eq[:, :group_q_length, :]  # (B, rq, D)
        Ec_g = Ec[:, :group_c_length, :]  # (B, rc, D)
        s = torch.sum(torch.max(torch.tensordot(Eq_g, Ec_g, dims=[[2], [2]]),  # (B, rq, B, rc)
                                dim=3)[0],  # (B, rq, B)
                      dim=1)  # (B, B)
        return s.flatten()

