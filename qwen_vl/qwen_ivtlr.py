import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
import logging
logging.basicConfig(
    filename='qwenvl_32_infer_sqa_time_epoch4.log',
    level=logging.DEBUG,         
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "loss_ce", "loss_hidden"])


class IVTLR(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        image_token_id,
        visual_start_id,
        visual_end_id,
        num_selected_patches: int = 32,
        disable_visual_insert: bool = False,
        disable_reasoning: bool = False,
    ):

        super(IVTLR, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.image_token_id = image_token_id
        self.visual_start_id = visual_start_id
        self.visual_end_id = visual_end_id
        self.num_selected_patches = num_selected_patches
        self.disable_visual_insert = disable_visual_insert
        self.disable_reasoning = disable_reasoning

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
        
    def forward(
        self,
        input_ids: torch.LongTensor,        # shape = (B, S)
        attention_mask: torch.LongTensor,    # shape = (B, S)
        labels: torch.LongTensor,            # shape = (B, S)
        position_ids: torch.LongTensor,      # shape = (B, S)
        pixel_values: torch.FloatTensor,     # shape = (B, 3, H, W)
        image_grid_thw: torch.Tensor = None,
        teacher_input_ids: torch.LongTensor = None,
        teacher_attention_mask: torch.LongTensor = None,
        teacher_position_ids: torch.LongTensor = None,
        teacher_sentence_end_positions: torch.LongTensor = None,
        teacher_alignment_mask: torch.Tensor = None,
        lambda_hidden: torch.Tensor = None,
        **kwargs
    ):

        B, S = input_ids.size()

        inputs_embeds = self.embedding(input_ids)  # (B, S, D)

        vs_indices = (input_ids == self.visual_start_id).nonzero(as_tuple=True)
        ve_indices = (input_ids == self.visual_end_id).nonzero(as_tuple=True)
        vs_pos_per_batch = {b.item(): vs_indices[1][i].item() for i, b in enumerate(vs_indices[0])}
        ve_pos_per_batch = {b.item(): ve_indices[1][i].item() for i, b in enumerate(ve_indices[0])}

        if pixel_values is not None:
            # Qwen-VL expects image features to be injected at image-token
            # positions before the language-model passes begin.
            pixel_values = pixel_values.type(self.base_causallm.visual.get_dtype())
            image_embeds = self.base_causallm.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_embeds.shape[0]}"
                )
            image_mask_init = (input_ids == self.image_token_id)  # (B, orig_S)
            expand_mask = image_mask_init.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1))
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(expand_mask, image_embeds)
        else:
            image_mask_init = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        

        image_mask = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        image_mask[:, :S] = image_mask_init


        for b in range(B):
            vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
            image_mask[b, vs+1:ve] = True

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == b]
            for b in range(B)
        ]
        if self.disable_reasoning:
            max_n_latents = 0
        else:
            max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        if max_n_latents > 0:
            first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0)
            end = first_latent_pos
        else:
            end = S
        
        kv_cache = None
        all_logits = []
        student_latent_states = []

        if max_n_latents > 0:
            for pass_idx in range(max_n_latents):
                start = 0
                hidden_states_offset = 0
                if kv_cache is None:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],  # (B, end, D)
                        attention_mask=attention_mask[:, start:end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )

                logits_this = outputs.logits                   
                hidden_states = outputs.hidden_states[-1]      
                attentions    = outputs.attentions              # list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)

                # Each latent token is filled with the hidden state immediately
                # before it. These same states are the student side of the
                # optional hidden-alignment loss.
                inputs_embeds_detached = inputs_embeds.detach().clone()
                pass_student_states = hidden_states.new_zeros((B, hidden_states.size(-1)))
                for b in range(B):
                    if len(latent_lists[b]) > pass_idx:
                        t_idx = latent_lists[b][pass_idx]
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        pass_student_states[b, :] = hidden_states[b, rel_pos, :]
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]

                student_latent_states.append(pass_student_states)
                inputs_embeds.data = inputs_embeds_detached

                if self.disable_visual_insert:
                    if pass_idx + 1 >= max_n_latents:
                        end = inputs_embeds.size(1)
                    else:
                        end = end + 1
                    continue

                # Use the current latent state attention to pick visual patches
                # that are inserted between latent reasoning steps. Insertions
                # change sequence length, so latent positions after `end` are
                # shifted by the number of selected visual embeddings.
                avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len)
                current_seq_len = avg_attn.size(1)
                select_image_embeds = []

                for b in range(B):
                    last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                    vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    scores = last_attn.clone()
                    allowed_positions = image_mask[b, :current_seq_len]  # shape=(S,)
                    invalid = ~allowed_positions
                    scores[invalid] = float("-inf")

                    rel_scores = scores[vs+1 : ve]  # (image_len,)
                    topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
                    abs_idxs = (vs + 1) + topk_rel
                    logging.debug(f"topk_rel: {topk_rel}")
                    logging.debug(f"abs idx: {abs_idxs}")
                    image_mask[b, abs_idxs] = False

                    picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    select_image_embeds.append(picked)

                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
                new_inputs_embeds = []
                new_attention_mask = []
                new_position_ids = []
                new_image_mask = []

                for b in range(B):
                    end_b = end
                    prefix_b = inputs_embeds[b, :end_b, :]    # (end_b, D)
                    suffix_b = inputs_embeds[b, end_b:, :]    # (old_len - end_b, D)
                    v_embed_b = select_image_embeds[b]       # (K, D)
                    merged_b = torch.cat([prefix_b, v_embed_b, suffix_b], dim=0)  # (old_len+K, D)
                    new_inputs_embeds.append(merged_b)

                    att_pref = attention_mask[b, :end_b]      # (end_b,)
                    att_suf  = attention_mask[b, end_b:]      # (old_len-end_b,)
                    att_v    = torch.ones(self.num_selected_patches, device=attention_mask.device, dtype=attention_mask.dtype)
                    merged_att = torch.cat([att_pref, att_v, att_suf], dim=0)  # (new_len,)
                    new_attention_mask.append(merged_att)

                    new_pos = torch.arange(merged_b.size(0), device=position_ids.device)
                    new_position_ids.append(new_pos)

                    img_pref = image_mask[b, :end_b]
                    img_suf  = image_mask[b, end_b:]
                    img_v    = torch.zeros(self.num_selected_patches, device=input_ids.device, dtype=torch.bool)
                    merged_img = torch.cat([img_pref, img_v, img_suf], dim=0)
                    new_image_mask.append(merged_img)

                padded_embeds = []
                padded_att   = []
                padded_pos   = []
                padded_img   = []

                for b in range(B):
                    emb_b = new_inputs_embeds[b]
                    att_b = new_attention_mask[b]
                    pos_b = new_position_ids[b]
                    img_b = new_image_mask[b]

                    padded_embeds.append(emb_b.unsqueeze(0))
                    padded_att.append(att_b.unsqueeze(0))
                    padded_pos.append(pos_b.unsqueeze(0))
                    padded_img.append(img_b.unsqueeze(0))

                inputs_embeds = torch.cat(padded_embeds, dim=0)    
                attention_mask = torch.cat(padded_att, dim=0)      
                position_ids    = torch.cat(padded_pos, dim=0)     
                image_mask     = torch.cat(padded_img, dim=0)   # (B, new_S)
                K = self.num_selected_patches
                for b in range(B):
                    for i, pos in enumerate(latent_lists[b]):
                        if pos > end:
                            latent_lists[b][i] = pos + K
                            logging.debug(f"latent pos: {latent_lists[b][i]}")

                if pass_idx + 1 >= max_n_latents:
                    end = inputs_embeds.size(1)
                else:
                    end = end + 1 + K

            if kv_cache:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            all_logits.append(outputs.logits)

        else:
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=False,
            )
            all_logits.append(outputs.logits)

        # `all_logits` contains the incremental latent passes plus the final
        # suffix pass. The original labels are right-aligned against this longer
        # logit stream because visual patch insertions add non-label positions.
        logits = torch.cat(all_logits, dim=-2)  # (B, total_len, V)
        B, final_S, V = logits.size()


        new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype)
        for b in range(B):
            num_labels = labels.size(1)
            new_labels[:, -num_labels:] = labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_hidden = loss_ce.new_tensor(0.0)

        if (
            lambda_hidden is not None
            and teacher_input_ids is not None
            and teacher_sentence_end_positions is not None
            and teacher_alignment_mask is not None
        ):
            if torch.is_tensor(lambda_hidden):
                lambda_hidden_value = lambda_hidden.to(device=loss_ce.device, dtype=loss_ce.dtype)
            else:
                lambda_hidden_value = loss_ce.new_tensor(float(lambda_hidden))

            teacher_alignment_mask = teacher_alignment_mask.to(device=input_ids.device, dtype=torch.bool)
            if (
                lambda_hidden_value.item() > 0
                and teacher_alignment_mask.any()
                and len(student_latent_states) > 0
            ):
                student_states = torch.stack(student_latent_states, dim=1)
                teacher_sentence_end_positions = teacher_sentence_end_positions.to(input_ids.device)

                # Teacher states come from the explicit-CoT sequence and are
                # detached by construction. This keeps CE as the only signal
                # updating the teacher path while aligning student latent states.
                with torch.no_grad():
                    teacher_outputs = self.base_causallm(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        position_ids=teacher_position_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=False,
                    )
                    teacher_hidden_states = teacher_outputs.hidden_states[-1]

                max_align_steps = min(
                    student_states.size(1),
                    teacher_sentence_end_positions.size(1),
                )
                if max_align_steps > 0:
                    active_mask = teacher_alignment_mask[:, :max_align_steps]
                    if active_mask.any():
                        gather_positions = teacher_sentence_end_positions[:, :max_align_steps].clamp(min=0)
                        gather_positions = gather_positions.unsqueeze(-1).expand(
                            -1, -1, teacher_hidden_states.size(-1)
                        )
                        teacher_states = torch.gather(
                            teacher_hidden_states,
                            dim=1,
                            index=gather_positions,
                        )
                        hidden_losses = 1.0 - F.cosine_similarity(
                            student_states[:, :max_align_steps, :].float(),
                            teacher_states.detach().float(),
                            dim=-1,
                        )
                        loss_hidden = hidden_losses[active_mask].mean().to(loss_ce.dtype)

            loss = loss_ce + lambda_hidden_value * loss_hidden
        else:
            loss = loss_ce

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            loss_ce=loss_ce,
            loss_hidden=loss_hidden,
        )


    def train(self, mode=True):
        self.base_causallm.train(mode)

    def eval(self):
        self.base_causallm.eval()
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            image_grid_thw: torch.Tensor = None,
            past_key_values: tuple = None,
            attention_mask: torch.Tensor = None,
            inputs_embeds: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            **kwargs
        ):
        
        self.base_causallm.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        pixel_values,
        image_grid_thw,
        max_new_tokens=16,
        output_embedding=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        eos_pos = None

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        
        current_ids = input_ids.clone()

        position_ids = torch.arange(
            0, current_ids.shape[1], 
            dtype=torch.long, 
            device=current_ids.device
        ).reshape(1, -1)

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            labels=current_ids.clone(),  
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )


        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
            

        current_inputs_embeds = outputs.inputs_embeds  # shape: (1, seq_len_after_insertion, hidden_dim)
        current_seq_len = current_inputs_embeds.shape[1]
        

        current_attention_mask = torch.ones((1, current_seq_len), device=current_inputs_embeds.device)
        

        next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

        self.gen_forward_cnt += 1
        

        past_key_values = None
        

        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                logging.debug(f"no kv_cache, using full embedding sequence")
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(
                        0, current_inputs_embeds.shape[1], 
                    dtype=torch.long, 
                        device=current_inputs_embeds.device
                ).reshape(1, -1)
            else:
                logging.debug(f"using kv_cache, input_shape: {next_token_embedding.shape}")
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor([[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device)

            outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None, 
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            
            next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

            self.gen_forward_cnt += 1

            if self.gen_forward_cnt % 10 == 0 and self.gen_forward_cnt >= 10:
                logging.debug(f"gen_forward_cnt: {self.gen_forward_cnt}")

            if next_token == self.eos_token_id:
                logging.debug(f"EOS token encountered at position {len(tokens)}, stopping generation")
                break

        print("generate 315")
        
        
        if output_embedding:
            return torch.tensor(tokens).view(1, -1), current_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)
