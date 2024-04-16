import seaborn as sns
from matplotlib import pyplot as plt
import random
import os, re
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch

from lm_debugger_utils import get_examples_df, get_examples_df_bert
import knockouts


def parse_args():
    def nullable_string(val):
        if val is None or val.lower() == 'none':
            return None
        return val

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path",
                        default=r"data/filtered_idioms.jsonl",
                        type=nullable_string,
                        required=False)
    parser.add_argument("--wiki_data_path",
                        default=r"data/gpt2-medium_random_wiki_examples.jsonl",
                        type=nullable_string,
                        required=False)
    parser.add_argument("--filters",
                        default=r"none",
                        type=nullable_string,
                        required=False)

    parser.add_argument("--out_dir",
                        default=r"data/output",
                        type=str,
                        required=False)

    parser.add_argument("--batch_size",
                        default=100,
                        type=int,
                        help="how many instances to run in parlale.",
                        required=False)

    parser.add_argument("--models",
                        default=r"bert-large-uncased,bert-base-uncased,gpt2-medium,gpt2-large",
                        type=str,
                        help="comma seperated list of the models, for now, only supports gpt2 models",
                        required=False)

    parser.add_argument("--pred_as_target",
                        default=False,
                        action='store_true',
                        required=False)

    parser.add_argument("--sample_size",
                        default=30,
                        type=int,
                        help="how many instances to sample for skip layer and knockouts, -1 for using all instances.",
                        required=False)

    parser.add_argument("--debug",
                        default=False,
                        action='store_true',
                        required=False)

    parser.add_argument("--run_knockout",
                        default=False,
                        action='store_true',
                        required=False)

    return parser.parse_args()


def get_examples_df_with_pred(debug, idioms_path, model, tokenizer, sample_size=-1,
                              start_sample_idx=0,use_first_subword_token=False,
                              lowercase=False,
                              filters=None,
                              save_keys=True,
                              bert_model=False):
    dfi = pd.read_json(idioms_path, lines=True)
    if 'prompt' not in dfi.columns:
        dfi['prompt'] = dfi['idiom']
    if lowercase:
        dfi['prompt'] = dfi['prompt'].str.lower()
    if debug:
        dfi = dfi[:10]
    idioms = dfi.prompt.values.tolist()
    random.shuffle(idioms)

    df = None
    if bert_model:
        df = get_examples_df_bert(idioms, model, tokenizer, target_first_subtoken=use_first_subword_token)
    else:
        df = get_examples_df(idioms, model, tokenizer, target_first_subtoken=use_first_subword_token,
                                    save_keys=save_keys)

        # df = pd.concat([df, df_tmp], axis=0, ignore_index=True)
        start_sample_idx += sample_size
    if not (filters is None):
        print(f"adding fiters {filters}")
        df = df.join(dfi[filters])
        if 'hard_to_guess' in dfi.columns:
            df['hard_to_guess'] = dfi['hard_to_guess']
    print(f"got  {len(df)} out of {len(dfi)} orig_idioms_len: {start_sample_idx} ")
    return df


def plot_prob_by_layer(df, model, out_name):
    df = df.copy()
    df_tmp = pd.DataFrame(df["layer_pred_probs"].to_list(), columns=[l for l in range(model.config.n_layer)])
    df_tmp[["data_legend"]] = df[["data_legend"]]

    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10, 5))
        tmp = pd.melt(df_tmp,
                      id_vars='data_legend',
                      var_name="layer",
                      value_name="pred_prob")
        ax = sns.lineplot(data=tmp, x="layer", y="pred_prob", hue="data_legend")
        ax.set_ylim(0, 1)
        ax.figure.savefig(f'{out_name}.pdf')


def plot_ranks_by_layer(df, model, out_name):
    df = df.copy()
    df_tmp = pd.DataFrame(df["layer_pred_ranks"].to_list(), columns=[l for l in range(model.config.n_layer)])
    df_tmp[["data_legend"]] = df[["data_legend"]]

    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10, 5))
        tmp = pd.melt(df_tmp,
                      id_vars='data_legend',
                      var_name="layer",
                      value_name="layer_pred_ranks")
        ax = sns.lineplot(data=tmp, x="layer", y="layer_pred_ranks", hue="data_legend")
        # ax.set_ylim(0, 1)
        ax.figure.savefig(f'{out_name}.pdf')


def plot_knockouts_heatmap(knock_df, model_name, dataset, dst):
    tabel_df = knock_df[~(knock_df['config'].isnull())][["config"]].groupby("config").agg("mean").reset_index()
    tabel_df[["start_layer", "end_layer", 'chosen_by']] = tabel_df.config.apply(
        lambda x: pd.Series([int(y) for y in x.split("_", 1)[0].strip("layers").split("-")] + x.split("_", 1)[1:])
    )

    def calc_changed_pred(row):
        exp = knock_df[knock_df['config'] == row['config']]
        res = (exp['pred'] != exp['target']).sum() / len(exp)
        return (exp['pred'] != exp['target']).sum() / len(exp)

    tabel_df['change_in_pred'] = tabel_df.apply(calc_changed_pred, axis=1)

    for by in set(tabel_df['chosen_by']):
        dfko_tmp = tabel_df[tabel_df['chosen_by'] == by]
        dfko_tmp = dfko_tmp.pivot(index='start_layer', columns='end_layer', values='change_in_pred')
        with sns.axes_style("whitegrid"):
            plt.figure(figsize=(24, 18))
            sns.heatmap(dfko_tmp, fmt="g", cmap='viridis',
                        cbar=False,
                        annot=True,
                        annot_kws={"size": 12}
                        )
            plt.title(by)
            print(f"saving heatmap to {dst}")
            plt.savefig(os.path.join(dst, f"{model_name}_{dataset}_knockouts_{by}.pdf"))


def plot_rank_and_prob(df, path, palette, num_layers):
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=(16, 12)
                             )
    ax = axes[0]
    df_tmp = pd.DataFrame(df["layer_pred_probs"].to_list(), columns=[l for l in range(num_layers)])
    df_tmp[["data_legend"]] = df[["data_legend"]]

    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(10, 5))
        tmp = pd.melt(df_tmp,
                      id_vars='data_legend',
                      var_name="layer",
                      value_name="probability")
        g1 = sns.lineplot(data=tmp, x="layer", y="probability",
                          palette=palette,
                          hue="data_legend",
                          ax=axes[0]
                          )

        g1.set(title='Candidate Promotion                       Confidence Boosting')
        df_tmp = pd.DataFrame(df["layer_pred_ranks"].to_list(), columns=[l for l in range(num_layers)])
        ax.set_ylim(0, 1)
        ax.legend_.set_title('')
        ax = axes[1]
        df_tmp[["data_legend"]] = df[["data_legend"]]
        tmp = pd.melt(df_tmp,
                      id_vars='data_legend',
                      var_name="layer",
                      value_name="rank")

        g2 = sns.lineplot(data=tmp, palette=palette, x="layer", y="rank", hue="data_legend", legend=False, ax=axes[1])
        xposition = [13]
        for xc in xposition:
            g1.axvline(x=xc, color='0.5', linestyle='--')
            g2.axvline(x=xc, color='0.5', linestyle='--')

        ax.set_xticks(range(0, num_layers, 4))
        ax.set_xticklabels([int(x + 1) for x in ax.get_xticks()])

        fig.align_labels()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)
        fig.savefig(path)
        return fig, axes


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = re.findall(r"[^, ]+", args.models)
    dataset_list = re.findall(r"[^, ]+", args.dataset_path)
    filters = re.findall(r"[^, ]+", args.filters) if args.filters else None
    for model_name in models:
        model_name = model_name.strip()
        os.makedirs(os.path.join(args.out_dir, model_name), exist_ok=True)
        wiki_cache_path = os.path.join(args.out_dir, model_name, "wiki_cache.pkl")
        if ('bert' in model_name):
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            num_layers = model.config.num_hidden_layers
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            num_layers = model.config.n_layer

        if os.path.exists(wiki_cache_path):
            wiki_df = pd.read_pickle(wiki_cache_path)
        else:
            wiki_df = get_examples_df_with_pred(debug=args.debug,
                                                idioms_path=args.wiki_data_path,
                                                model=model,
                                                tokenizer=tokenizer,
                                                use_first_subword_token=True,
                                                sample_size=args.batch_size,
                                                save_keys=False,
                                                bert_model=('bert' in model_name)
                                                )
            wiki_df.to_pickle(wiki_cache_path)

        wiki_df['subset'] = "wiki"
        df_list = [wiki_df]
        for dataset in dataset_list:
            if torch.cuda.is_available():
                model.cuda()
            model.eval()
            dataset_name = re.search("(\w+)\.\w+$", dataset).group(1)
            path = os.path.join(args.out_dir, model_name, dataset_name)
            os.makedirs(path, exist_ok=True)
            dataset_out_path = os.path.join(path, f"{model_name}_{dataset_name}_plot_data.pkl")
            dataset_beckup_out_path = os.path.join(path, f"{model_name}_{dataset_name}_beckup_data.pkl")
            correct_idioms_out_path = os.path.join(path, f"{model_name}_{dataset_name}_correct.pkl")
            incorrect_idioms_out_path = os.path.join(path, f"{model_name}_{dataset_name}_incorrect.pkl")
            knockouts_out_path = os.path.join(path, f"{model_name}_{dataset_name}_knockouts.pkl")
            plot_out_path = os.path.join(path, f"{model_name}_{filters}_")

            df = get_examples_df_with_pred(debug=args.debug,
                                           idioms_path=dataset,
                                           model=model,
                                           tokenizer=tokenizer,
                                           use_first_subword_token=True,
                                           sample_size=args.batch_size,
                                           filters=filters,
                                           save_keys=False,
                                           bert_model=('bert' in model_name)
                                           )
            print(f"saving for backup all data to {path}")
            df.to_pickle(dataset_beckup_out_path)
            # if dataset in datasets_to_split:
            is_memorized = df['pred'] == df['target']

            if filters:
                is_filtered = df[filters].all(axis=1)

            df['is_memorized'] = is_memorized
            memorized_idioms_df = df[is_memorized]
            print(" {} memorized: {} out of {}".format(model_name, len(memorized_idioms_df), len(df)))
            df['subset'] = df.apply(
                (lambda row: 'mem-idiom' if row['is_memorized'] else "unmem-idiom")
                , axis=1)

            if args.run_knockout:
                res = knockouts.main(model, tokenizer, memorized_idioms_df,
                                     sample_size=args.sample_size,
                                     output_df_path=knockouts_out_path,
                                     pred_as_target=args.pred_as_target)

            df_list.append(df)
            df_all = pd.concat(df_list, axis=0, ignore_index=True)
            legend_map = {
                "mem-idiom": 'Memorized',
                "unmem-idiom": "Unmemorized",
                "wiki": "Wikipedia",
            }
            palette = {
                "Wikipedia": '#E66100',
                "Mem-idiom": "#D81B60",  # 'bluish green' ,
                "Unmem-idiom": '#009E73'  # "#a81839" #'reddish purple'
            }
            df_all["data_legend"] = df_all["subset"].apply(legend_map.get)
            print(f"saving all data to {dataset_out_path}")
            df_all.to_pickle(dataset_out_path)
