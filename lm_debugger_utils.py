import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

"""
This code was adapted from https://github.com/mega002/lm-debugger
"""

def set_hooks_bert(model):
    """
    Only works on bert from HF
    """

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(layer):
        def hook(module, input, output):
            model.activations_[f"layer_{layer}"] = output[0][0][-3].detach()
        return hook

    for i in range(model.config.num_hidden_layers):
        model.base_model.encoder.layer[i].register_forward_hook(get_activation(i))


def set_hooks_gpt2(model):
    """
    Only works on GPT2 from HF
    """
    final_layer = model.config.n_layer - 1

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name or "m_tag_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    model.activations_[name] = output[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[0]  # [num_tokens, 3072] for values;
                    model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    model.activations_[name] = input[0][:, num_tokens - 1].detach()
                elif "m_tag_coef" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    model.activations_[name] = output[0][num_tokens - 1].detach()

            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                if name == "layer_residual_" + str(final_layer):
                    model.activations_[name] = model.activations_["intermediate_residual_" + str(final_layer)] + \
                                               model.activations_["mlp_" + str(final_layer)]
                else:
                    model.activations_[name] = input[0][:,
                                               num_tokens - 1].detach()  # https://github.com/huggingface/transformers/issues/7760

        return hook

    model.transformer.h[0].ln_1.register_forward_hook(get_activation("input_embedding"))

    for i in range(model.config.n_layer):
        if i != 0:
            model.transformer.h[i].ln_1.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
        model.transformer.h[i].ln_2.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

        model.transformer.h[i].attn.register_forward_hook(get_activation("attn_" + str(i)))
        model.transformer.h[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
        model.transformer.h[i].mlp.c_proj.register_forward_hook(get_activation("m_coef_" + str(i)))
        model.transformer.h[i].mlp.c_fc.register_forward_hook(get_activation("m_tag_coef_" + str(i)))

    model.transformer.ln_f.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))


def set_control_hooks_gpt2(model, values_per_layer, coef_value=0):
    def change_values(values, coef_val):
        def hook(module, input, output):
            output[:, :, values] = coef_val

        return hook

    hooks = []
    for l in range(model.config.n_layer):
        if l in values_per_layer:
            values = values_per_layer[l]
        else:
            values = []
        hook = model.transformer.h[l].mlp.c_fc.register_forward_hook(
            change_values(values, coef_value)
        )
        hooks.append(hook)

    return hooks


def set_layer_skip_hooks_gpt2(model, layers_to_skip):
    def skip_layer_hook():
        def hook(module, input, output):
            # IMPORTANT! we only replace the output hidden states with the input hidden states,
            # without replacing the other extra outputs. See more details here:
            # https://github.com/huggingface/transformers/blob/cd583bdaa543318785cc2a74abb195546d972a25/src/transformers/models/gpt2/modeling_gpt2.py#L441
            return (input[0],) + output[1:]

        return hook

    hooks = []
    for l in layers_to_skip:
        hook = model.transformer.h[l].register_forward_hook(
            skip_layer_hook()
        )
        hooks.append(hook)

    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_resid_predictions_bert(model, tokenizer, sentence, TOP_K=1,
                          start_idx=None, end_idx=None, target_token=None):
    HIDDEN_SIZE = model.config.hidden_size

    layer_residual_preds = []
    layer_residual_pred_ranks = []
    layer_residual_pred_probs = []
    layer_residual_target_ranks = []
    layer_residual_target_probs = []

    if start_idx is not None and end_idx is not None:
        tokens = [
            token for token in sentence.split(' ')
            if token not in ['', '\n']
        ]

        sentence = " ".join(tokens[start_idx:end_idx])

    tokens = tokenizer(sentence, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens.to(device)
    with torch.no_grad():
        output = model(**tokens)
    pred_token = output['logits'][0][-3].argmax().item()

    for layer in model.activations_.keys():
        logits = model.cls(model.activations_[layer])

        probs = F.softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()

        # assert probs add to 1
        assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

        probs_argsort_desc = np.argsort(probs)[::-1]

        top_k = [(probs[t], tokenizer.decode(t)) for t in probs_argsort_desc[:TOP_K]]
        pred_rank = np.where(probs_argsort_desc == pred_token)[0][0]
        pred_prob = probs[pred_token]
        pred_logit = logits[pred_token].item()

        if target_token is not None:
            target_rank = np.where(probs_argsort_desc == target_token)[0][0]
            target_prob = probs[target_token]
            target_logit = logits[target_token].item()
        else:
            target_rank = -1
            target_prob = -1
            target_logit = -1

        layer_residual_preds.append(top_k)
        layer_residual_pred_ranks.append(pred_rank)
        layer_residual_pred_probs.append(pred_prob)
        layer_residual_target_ranks.append(target_rank)
        layer_residual_target_probs.append(target_prob)


        for attr in ["layer_resid_preds",
                     "layer_resid_pred_ranks",
                     "layer_resid_pred_probs",
                     "layer_resid_target_ranks",
                     "layer_resid_target_probs",]:
            if not hasattr(model, attr):
                setattr(model, attr, [])

        model.pred_token = pred_token

        model.layer_resid_preds = layer_residual_preds
        model.layer_resid_pred_ranks = layer_residual_pred_ranks
        model.layer_resid_pred_probs = layer_residual_pred_probs
        model.layer_resid_target_ranks = layer_residual_target_ranks
        model.layer_resid_target_probs = layer_residual_target_probs


def get_resid_predictions(model, tokenizer, sentence, TOP_K=1,
                          start_idx=None, end_idx=None, target_token=None):
    HIDDEN_SIZE = model.config.n_embd

    layer_residual_preds = []
    intermed_residual_preds = []
    mlp_preds = []

    layer_residual_pred_ranks = []
    intermed_residual_pred_ranks = []
    mlp_pred_ranks = []

    layer_residual_pred_probs = []
    intermed_residual_pred_probs = []
    mlp_pred_scores = []

    layer_residual_target_ranks = []
    intermed_residual_target_ranks = []
    mlp_target_ranks = []

    layer_residual_target_probs = []
    intermed_residual_target_probs = []
    mlp_target_scores = []

    if start_idx is not None and end_idx is not None:
        tokens = [
            token for token in sentence.split(' ')
            if token not in ['', '\n']
        ]

        sentence = " ".join(tokens[start_idx:end_idx])

    tokens = tokenizer(sentence, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens.to(device)

    output = model(**tokens, output_hidden_states=True)
    pred_token = output['logits'][0][-1].argmax().item()

    for layer in model.activations_.keys():
        if "layer_residual" in layer or "intermediate_residual" in layer or "mlp" in layer:
            if len(model.activations_[layer].shape) == 1:
                normed = model.transformer.ln_f(torch.unsqueeze(model.activations_[layer], 0))
            else:
                normed = model.transformer.ln_f(model.activations_[layer])

            logits = torch.matmul(model.lm_head.weight, normed.T)

            probs = F.softmax(logits.T[0], dim=-1)
            probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

            # assert probs add to 1
            assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

            probs_argsort_desc = np.argsort(probs)[::-1]

            top_k = [(probs[t], tokenizer.decode(t)) for t in probs_argsort_desc[:TOP_K]]
            pred_rank = np.where(probs_argsort_desc == pred_token)[0][0]
            pred_prob = probs[pred_token]
            pred_logit = logits.T[0][pred_token].item()

            if target_token is not None:
                target_rank = np.where(probs_argsort_desc == target_token)[0][0]
                target_prob = probs[target_token]
                target_logit = logits.T[0][target_token].item()
            else:
                target_rank = -1
                target_prob = -1
                target_logit = -1

        if "layer_residual" in layer:
            layer_residual_preds.append(top_k)
            layer_residual_pred_ranks.append(pred_rank)
            layer_residual_pred_probs.append(pred_prob)
            layer_residual_target_ranks.append(target_rank)
            layer_residual_target_probs.append(target_prob)
        elif "intermediate_residual" in layer:
            intermed_residual_preds.append(top_k)
            intermed_residual_pred_ranks.append(pred_rank)
            intermed_residual_pred_probs.append(pred_prob)
            intermed_residual_target_ranks.append(target_rank)
            intermed_residual_target_probs.append(target_prob)
        elif "mlp" in layer:
            mlp_preds.append(top_k)
            mlp_pred_ranks.append(pred_rank)
            mlp_target_ranks.append(target_rank)
            mlp_pred_scores.append(pred_logit)
            mlp_target_scores.append(target_logit)

        for attr in ["layer_resid_preds", "intermed_residual_preds", "mlp_preds",
                     "layer_resid_pred_ranks", "intermed_residual_pred_ranks", "mlp_pred_ranks",
                     "layer_resid_pred_probs", "intermed_residual_pred_probs", "mlp_pred_scores",
                     "layer_resid_target_ranks", "intermed_residual_target_ranks", "mlp_target_ranks",
                     "layer_resid_target_probs", "intermed_residual_target_probs", "mlp_target_scores"]:
            if not hasattr(model, attr):
                setattr(model, attr, [])

        model.pred_token = pred_token

        model.layer_resid_preds = layer_residual_preds
        model.intermed_residual_preds = intermed_residual_preds
        model.mlp_preds = mlp_preds

        model.layer_resid_pred_ranks = layer_residual_pred_ranks
        model.intermed_residual_pred_ranks = intermed_residual_pred_ranks
        model.mlp_pred_ranks = mlp_pred_ranks

        model.layer_resid_pred_probs = layer_residual_pred_probs
        model.intermed_residual_pred_probs = intermed_residual_pred_probs
        model.mlp_pred_scores = mlp_pred_scores

        model.layer_resid_target_ranks = layer_residual_target_ranks
        model.intermed_residual_target_ranks = intermed_residual_target_ranks
        model.mlp_target_ranks = mlp_target_ranks

        model.layer_resid_target_probs = layer_residual_target_probs
        model.intermed_residual_target_probs = intermed_residual_target_probs
        model.mlp_target_scores = mlp_target_scores


def project_value_to_vocab(model, tokenizer, layer, value_idx, top_k=10):
    normed = model.transformer.ln_f(model.transformer.h[layer].mlp.c_proj.weight.data[value_idx])

    logits = torch.matmul(model.lm_head.weight, normed.T)
    probs = F.softmax(logits, dim=-1)
    probs = torch.reshape(probs, (-1,)).detach().numpy()

    probs_ = []
    for index, prob in enumerate(probs):
        probs_.append((index, prob))

    top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
    value_preds = [(tokenizer.decode(t[0]), t[0]) for t in top_k]

    return value_preds


def get_preds_and_hidden_states_bert(prompts, bert_model, bert_tokenizer,
                                target_tokens=None, knockouts=None):
    set_hooks_bert(bert_model)

    sent_to_preds = {}
    sent_to_hidden_states = {}

    for prompt_i, prompt in tqdm(enumerate(prompts)):
        if target_tokens is not None:
            target_token = target_tokens[prompt]    # [prompt_i]
        else:
            target_token = None

        # # add knockout hooks
        # if knockouts is not None:
        #     hooks = set_control_hooks_gpt2(bert_model, knockouts[prompt])
        # else:
        #     hooks = []

        get_resid_predictions_bert(bert_model, bert_tokenizer, prompt, TOP_K=10, target_token=target_token)

        # remove knockout hook
        # remove_hooks(hooks)

        if prompt not in sent_to_preds.keys():
            sent_to_preds[prompt] = {}

        sent_to_preds[prompt]["pred_token"] = bert_model.pred_token

        sent_to_preds[prompt]["layer_resid_preds"] = bert_model.layer_resid_preds
        sent_to_preds[prompt]["layer_resid_pred_ranks"] = bert_model.layer_resid_pred_ranks
        sent_to_preds[prompt]["layer_resid_pred_probs"] = bert_model.layer_resid_pred_probs
        sent_to_preds[prompt]["layer_resid_target_ranks"] = bert_model.layer_resid_target_ranks
        sent_to_preds[prompt]["layer_resid_target_probs"] = bert_model.layer_resid_target_probs
        sent_to_hidden_states[prompt] = bert_model.activations_.copy()

    return sent_to_hidden_states, sent_to_preds


def get_preds_and_hidden_states(prompts, gpt2_model, gpt2_tokenizer,
                                target_tokens=None, knockouts=None):
    set_hooks_gpt2(gpt2_model)

    sent_to_preds = {}
    sent_to_hidden_states = {}

    for prompt_i, prompt in tqdm(enumerate(prompts)):
        if target_tokens is not None:
            target_token = target_tokens[prompt]    # [prompt_i]
        else:
            target_token = None

        # add knockout hooks
        if knockouts is not None:
            hooks = set_control_hooks_gpt2(gpt2_model, knockouts[prompt])
        else:
            hooks = []

        get_resid_predictions(gpt2_model, gpt2_tokenizer, prompt, TOP_K=10, target_token=target_token)

        # remove knockout hook
        remove_hooks(hooks)

        if prompt not in sent_to_preds.keys():
            sent_to_preds[prompt] = {}

        sent_to_preds[prompt]["pred_token"] = gpt2_model.pred_token

        sent_to_preds[prompt]["layer_resid_preds"] = gpt2_model.layer_resid_preds
        sent_to_preds[prompt]["intermed_residual_preds"] = gpt2_model.intermed_residual_preds
        sent_to_preds[prompt]["mlp_preds"] = gpt2_model.mlp_preds

        sent_to_preds[prompt]["layer_resid_pred_ranks"] = gpt2_model.layer_resid_pred_ranks
        sent_to_preds[prompt]["intermed_residual_pred_ranks"] = gpt2_model.intermed_residual_pred_ranks
        sent_to_preds[prompt]["mlp_pred_ranks"] = gpt2_model.mlp_pred_ranks

        sent_to_preds[prompt]["layer_resid_pred_probs"] = gpt2_model.layer_resid_pred_probs
        sent_to_preds[prompt]["intermed_residual_pred_probs"] = gpt2_model.intermed_residual_pred_probs
        sent_to_preds[prompt]["mlp_pred_scores"] = gpt2_model.mlp_pred_scores

        sent_to_preds[prompt]["layer_resid_target_ranks"] = gpt2_model.layer_resid_target_ranks
        sent_to_preds[prompt]["intermed_residual_target_ranks"] = gpt2_model.intermed_residual_target_ranks
        sent_to_preds[prompt]["mlp_target_ranks"] = gpt2_model.mlp_target_ranks

        sent_to_preds[prompt]["layer_resid_target_probs"] = gpt2_model.layer_resid_target_probs
        sent_to_preds[prompt]["intermed_residual_target_probs"] = gpt2_model.intermed_residual_target_probs
        sent_to_preds[prompt]["mlp_target_scores"] = gpt2_model.mlp_target_scores

        sent_to_hidden_states[prompt] = gpt2_model.activations_.copy()

    return sent_to_hidden_states, sent_to_preds

FILTER_SUBWORDS= False


def process_preds_and_hidden_states_bert(sent_to_hidden_states, sent_to_preds,
                                    model, tokenizer, top_k, targets_info=None):
    records = []
    for sent_i, sent in tqdm(enumerate(sent_to_preds.keys())):
        layer_preds_probs = []
        layer_preds_tokens = []

        for k in sent_to_preds[sent]['layer_resid_preds']:
            layer_p_probs, layer_p_tokens = zip(*k)
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)

        record = {
            "prompt": sent,
            "pred": tokenizer.decode(sent_to_preds[sent]['pred_token']).replace(' ', ''),
            "pred_token": sent_to_preds[sent]['pred_token'],
            "layer_target_ranks": sent_to_preds[sent]['layer_resid_target_ranks'],
            "layer_target_probs": sent_to_preds[sent]['layer_resid_target_probs'],
            "layer_pred_ranks": sent_to_preds[sent]['layer_resid_pred_ranks'],
            "layer_pred_probs": sent_to_preds[sent]['layer_resid_pred_probs'],
            "layer_preds_probs": layer_preds_probs,
            "layer_preds_tokens": layer_preds_tokens,
        }
        record.update({
                "target": targets_info["targets"][sent],
                "target_token": targets_info["target_tokens"][sent],
                "is_subword": not targets_info["targets"][sent].startswith(' ')
            })
        record.update(targets_info)

        records.append(record)

    return records


def process_preds_and_hidden_states(sent_to_hidden_states, sent_to_preds,
                                    model, tokenizer, top_k, targets_info=None ,save_keys = True ):
    records = []
    for sent_i, sent in tqdm(enumerate(sent_to_preds.keys())):
        top_coef_idx = []
        top_coef_vals = []
        residual_preds_probs = []
        residual_preds_tokens = []
        layer_preds_probs = []
        layer_preds_tokens = []
        mlp_preds_probs = []
        mlp_preds_tokens = []
        m_coefs_all = []
        m_tag_coefs_all = []

        keys = model.state_dict()[f'transformer.h.1.mlp.c_fc.weight'].cpu().numpy()
        keys = np.linalg.norm(keys, axis=0)
        for LAYER in range(model.config.n_layer):
            coefs_ = []
            m_coefs = sent_to_hidden_states[sent]["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
            if save_keys:
                m_coefs_all.append(m_coefs/keys)
                m_tag_coefs = sent_to_hidden_states[sent]["m_tag_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
                m_tag_coefs_all.append(m_tag_coefs)

            value_norms = torch.linalg.norm(model.transformer.h[LAYER].mlp.c_proj.weight.data, dim=1)
            scaled_coefs = np.absolute(m_coefs) * value_norms.cpu().numpy()
            for index, prob in enumerate(scaled_coefs):
                coefs_.append((index, prob))

            top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:top_k]
            c_idx, c_vals = zip(*top_values)
            top_coef_idx.append(c_idx)
            top_coef_vals.append(c_vals)

            residual_p_probs, residual_p_tokens = zip(*sent_to_preds[sent]['intermed_residual_preds'][LAYER])
            residual_preds_probs.append(residual_p_probs)
            residual_preds_tokens.append(residual_p_tokens)

            layer_p_probs, layer_p_tokens = zip(*sent_to_preds[sent]['layer_resid_preds'][LAYER])
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)

            mlp_p_probs, mlp_p_tokens = zip(*sent_to_preds[sent]['mlp_preds'][LAYER])
            mlp_preds_probs.append(mlp_p_probs)
            mlp_preds_tokens.append(mlp_p_tokens)

        record = {
            "prompt": sent,
            "pred": tokenizer.decode(sent_to_preds[sent]['pred_token']),
            "pred_token": sent_to_preds[sent]['pred_token'],
            "residual_target_ranks": sent_to_preds[sent]['intermed_residual_target_ranks'],
            "layer_target_ranks": sent_to_preds[sent]['layer_resid_target_ranks'],
            "mlp_target_ranks": sent_to_preds[sent]['mlp_target_ranks'],
            "residual_target_probs": sent_to_preds[sent]['intermed_residual_target_probs'],
            "layer_target_probs": sent_to_preds[sent]['layer_resid_target_probs'],
            "mlp_target_scores": sent_to_preds[sent]['mlp_target_scores'],
            "residual_pred_ranks": sent_to_preds[sent]['intermed_residual_pred_ranks'],
            "layer_pred_ranks": sent_to_preds[sent]['layer_resid_pred_ranks'],
            "mlp_pred_ranks": sent_to_preds[sent]['mlp_pred_ranks'],
            "residual_pred_probs": sent_to_preds[sent]['intermed_residual_pred_probs'],
            "layer_pred_probs": sent_to_preds[sent]['layer_resid_pred_probs'],
            "mlp_pred_scores": sent_to_preds[sent]['mlp_pred_scores'],
            "top_coef_idx": top_coef_idx,
            "top_coef_vals": top_coef_vals,
            "residual_preds_probs": residual_preds_probs,
            "residual_preds_tokens": residual_preds_tokens,
            "layer_preds_probs": layer_preds_probs,
            "layer_preds_tokens": layer_preds_tokens,
            "mlp_preds_probs": mlp_preds_probs,
            "mlp_preds_tokens": mlp_preds_tokens,

        }
        if save_keys:
            record["m_coefs"] = np.stack(m_coefs_all)
            record["m_tag_coefs"] = np.stack(m_tag_coefs_all)
        if targets_info is not None:
            if FILTER_SUBWORDS and not targets_info["targets"][sent].startswith(' '):
                continue

            record.update({
                "target": targets_info["targets"][sent],
                "target_token": targets_info["target_tokens"][sent],
                "is_subword": not targets_info["targets"][sent].startswith(' ')
            })
            if "full_targets" in targets_info.keys():
                record["full_target"] = targets_info["full_targets"][sent]

        records.append(record)

    return records


def get_examples_df_bert(idioms, model, tokenizer, top_k=100, target_first_subtoken=False):
    prompts = []
    targets = {}
    target_tokens = {}
    full_targets = {}

    for idiom in idioms:
        seq = idiom.split(' ')
        full_target = seq[-1]
        prompt = ' '.join(seq[:-1]) + ' [MASK].'
        target_token = tokenizer(full_target)['input_ids'][1]
        target = tokenizer.decode(target_token).replace(' ', '')

        prompts.append(prompt)

        targets[prompt] = target
        target_tokens[prompt] = target_token
        full_targets[prompt] = target

    sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states_bert(prompts, model, tokenizer,
                                                                       target_tokens=target_tokens)

    targets_info = {"targets": targets, "target_tokens": target_tokens, "full_targets": full_targets}
    records = process_preds_and_hidden_states_bert(sent_to_hidden_states, sent_to_preds,
                                              model, tokenizer, top_k, targets_info)

    df = pd.DataFrame(records)
    return df


def get_examples_df(idioms, model, tokenizer, top_k=100, target_first_subtoken=False,save_keys = True):
    prompts = []
    targets = {}
    target_tokens = {}
    full_targets = {}
    for tokens in tokenizer(idioms)['input_ids']:
        token_idx = -1

        if target_first_subtoken:
            while not tokenizer.decode(tokens[token_idx]).startswith(' '):
                token_idx -= 1

        prompt = tokenizer.decode(tokens[:token_idx])
        prompts.append(prompt)

        targets[prompt] = tokenizer.decode(tokens[token_idx])
        target_tokens[prompt] = tokens[token_idx]
        full_targets[prompt] = tokenizer.decode(tokens[token_idx:])

    sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states(prompts, model, tokenizer,
                                                                       target_tokens=target_tokens)

    targets_info = {"targets": targets, "target_tokens": target_tokens, "full_targets": full_targets}
    records = process_preds_and_hidden_states(sent_to_hidden_states, sent_to_preds,
                                              model, tokenizer, top_k, targets_info, save_keys = save_keys)

    df = pd.DataFrame(records)
    return df


def get_examples_df_for_prompts(prompts, targets_info, model, tokenizer,
                                top_k=100, knockout_config=None,save_keys = True):
    sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states(prompts, model, tokenizer,
                                                                       target_tokens=targets_info["target_tokens"],
                                                                       knockouts=knockout_config)

    records = process_preds_and_hidden_states(sent_to_hidden_states, sent_to_preds,
                                              model, tokenizer, top_k, targets_info,save_keys = save_keys)

    df = pd.DataFrame(records)
    return df


def get_all_projected_value_vectors(model):
    logits = []
    for i in tqdm(range(model.config.n_layer)):
        layer_logits = torch.matmul(model.transformer.wte.weight, model.transformer.h[i].mlp.c_proj.weight.T).T
        logits.append(layer_logits)

    logits = torch.vstack(logits)
    return logits.detach().cpu().numpy()


def agg_score_in_top_values_per_layer(projs, layer_fc2_vals, row, ks):
    results = {
        k: [[], [], [], [], [], []]
        for k in ks
    }
    max_k = max(ks)

    pred_token = row["pred_token"]
    for layer, dims in enumerate(row["top_coef_idx"]):
        layer_pred_scores = []
        layer_pred_scores_norm = []
        v_norms = torch.linalg.norm(layer_fc2_vals[layer].T[list(dims[:max_k])], dim=1)
        for dim_i, dim in enumerate(dims[:max_k]):
            ve = projs[layer, dim, pred_token]
            mv_norm = row["top_coef_vals"][layer][dim_i]
            m = mv_norm / v_norms[dim_i]
            mve = m * ve
            layer_pred_scores.append(mve)

            vE = projs[layer, dim]
            vE_norm = vE / np.linalg.norm(vE)
            ve_norm = vE_norm[pred_token]
            mve_norm = m * ve_norm
            layer_pred_scores_norm.append(mve_norm)

        for k in ks:
            results[k][0].append(np.min(layer_pred_scores[:k]))
            results[k][1].append(np.mean(layer_pred_scores[:k]))
            results[k][2].append(np.max(layer_pred_scores[:k]))
            results[k][3].append(np.min(layer_pred_scores_norm[:k]))
            results[k][4].append(np.mean(layer_pred_scores_norm[:k]))
            results[k][5].append(np.max(layer_pred_scores_norm[:k]))

    results_lists = []
    for k in ks:
        results_lists.extend(results[k])

    return results_lists


def top_vv_stats(projs_arg, projs_per_l, row, tokenizer, debug):
    results = [[], [], []]
    pred_token = row["pred_token"]
    for layer, dims in enumerate(row["top_coef_idx"]):
        top_vv_dim = dims[0]
        pred_rank_in_top_value_vector = np.where(projs_arg[(layer, top_vv_dim)] == pred_token)[0][0]
        results[0].append(pred_rank_in_top_value_vector)

        top_value_vector_10tokens = [tokenizer.decode(projs_arg[(layer, top_vv_dim)][i]) for i in range(10)]
        results[1].append(top_value_vector_10tokens)

        top_value_vector_near_pred_10tokens = [tokenizer.decode(projs_arg[(layer, top_vv_dim)][i]) for i in
                                               range(max(0, pred_rank_in_top_value_vector-5),
                                                     min(len(projs_arg[(0, 0)]), pred_rank_in_top_value_vector+5))]
        results[2].append(top_value_vector_near_pred_10tokens)

        if debug and layer > 0:
            break

    return results #[[pred_rank_in_top_value_vector],[top_value_vector_10tokens],[top_value_vector_near_pred_10tokens]]


def agg_rank_in_top_values_per_layer(projs_arg, row, ks):
    results = {
        k: [[], [], []]
        for k in ks
    }
    max_k = max(ks)
    pred_token = row["pred_token"]
    for layer, dims in enumerate(row["top_coef_idx"]):
        layer_pred_scores = [
            np.where(projs_arg[(layer, dim)] == pred_token)[0][0]
            for dim in dims[:max_k]
        ]

        for k in ks:
            results[k][0].append(np.min(layer_pred_scores[:k]))
            results[k][1].append(np.mean(layer_pred_scores[:k]))
            results[k][2].append(np.max(layer_pred_scores[:k]))

    results_lists = []
    for k in ks:
        results_lists.extend(results[k])

    return results_lists


def top_promoting_values_ranks(pred_token, k, n_layer, projs_per_l):
    top_promoting_values = []
    top_promoting_values_ranks = []
    for l in range(n_layer):
        pred_token_ranks = np.where(projs_per_l[l] == pred_token)[1]
        pred_token_ranks_smallest = np.argpartition(pred_token_ranks, k)[:k]
        pred_token_ranks_smallest_vals = [pred_token_ranks[dim] for dim in pred_token_ranks_smallest]
        # TODO(mega): sort them!

        top_promoting_values.append(pred_token_ranks_smallest)
        top_promoting_values_ranks.append(pred_token_ranks_smallest_vals)

    return top_promoting_values, top_promoting_values_ranks


# in debug we only load 2 layers
def get_model_static_files(model, projs, debug):
    projs_ = projs.reshape((model.config.n_layer, model.transformer.h[0].mlp.c_proj.weight.size(0), projs.shape[1]))
    if debug:
        projs_ = projs_[:2]

    d = {}
    inv_d = {}
    cnt = 0
    total_dims = model.transformer.h[0].mlp.c_proj.weight.size(0)
    for i in range(model.config.n_layer):
        for j in range(total_dims):
            d[cnt] = (i, j)
            inv_d[(i, j)] = cnt
            cnt += 1
        if debug and i > 0:
            break

    projs_arg = {}
    for i in tqdm(range(model.config.n_layer)):
        for j in tqdm(range(total_dims), leave=False):
            k = (i, j)
            cnt = inv_d[(i, j)]
            ids = np.argsort(-projs[cnt])
            projs_arg[k] = ids
        if debug and i > 0:
            break

    projs_per_l = {}
    for i in tqdm(range(model.config.n_layer)):
        projs_per_l[i] = np.vstack([
            projs_[(i, j)]
            for j in range(total_dims)
        ])
        if debug and i > 0:
            break

    layer_fc2_vals = [
        model.transformer.h[layer_i].mlp.c_proj.weight.T.detach()
        for layer_i in tqdm(range(model.config.n_layer))
    ]
    if debug:
        layer_fc2_vals = layer_fc2_vals[:2]

    return projs_, projs_arg, projs_per_l, layer_fc2_vals


def merge_with_projected_value_vectors(df, model, tokenizer, projs_, projs_arg, projs_per_l, layer_fc2_vals,
                                       debug=False):
    df_temp = df.apply(
        lambda row: pd.Series(top_vv_stats(projs_arg, projs_per_l, row, tokenizer, debug)),
        axis=1
    )

    # df_temp: [[pred_rank_in_top_value_vector], [top_value_vector_10tokens], [top_value_vector_near_pred_10tokens]]

    df["top1vv_pred_rank"] = df_temp[0]
    df["top1vv_tokens"] = df_temp[1]
    df["top1vv_near_pred_tokens"] = df_temp[2]

    ks = [1, 5, 10, 50, 100]
    score_cols = []
    rank_cols = []
    for k in ks:
        score_cols.extend([f"min_score_in_top_values{k}", f"mean_score_in_top_values{k}", f"max_score_in_top_values{k}",
                           f"min_score_in_top_values{k}_norm", f"mean_score_in_top_values{k}_norm",
                           f"max_score_in_top_values{k}_norm"])
        rank_cols.extend([f"min_rank_in_top_values{k}", f"mean_rank_in_top_values{k}", f"max_rank_in_top_values{k}"])

    df[score_cols] = df.apply(
        lambda row: pd.Series(agg_score_in_top_values_per_layer(projs_, layer_fc2_vals, row, ks)),
        axis=1
    )
    df[rank_cols] = df.apply(
        lambda row: pd.Series(agg_rank_in_top_values_per_layer(projs_arg, row, ks)),
        axis=1
    )

    return df
