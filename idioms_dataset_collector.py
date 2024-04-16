
import argparse
import json
import numpy as np
import pandas as pd
import secrets
import spacy
import tqdm
import random

from rdflib import Graph

nlp = spacy.load("en_core_web_sm")
trim_characters = tuple(['.','?',' ',"!"])


# function to get unique values
def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def _clean_idiom(idiom):
    idiom = idiom.lower()
    while idiom.endswith(trim_characters) and len(idiom) > 0:
        idiom = idiom[:-1]

    idiom = idiom.replace("\n", "").replace("\t", "")
    idiom = idiom.strip('\"')

    return idiom


def is_prefix_stopwords(idiom):
    parsed = nlp(idiom)
    assert len(parsed) > 1
    prefix = parsed[:-1]
    return sum([x.is_stop for x in prefix]) == len(prefix)


def read_magpie(path, min_length):
    df = pd.read_json(path, lines=True)
    
    # filter by length & idiomatic/literal usage
    df["idiom_length"] = df.idiom.apply(lambda x: len(x.split(" ")))
    df = df[(df.idiom_length > min_length) & (df.label == "i")][["idiom"]].drop_duplicates()
    
    # filter by stopwords
    df["is_prefix_stopwords"] = df.idiom.apply(lambda x: is_prefix_stopwords(x))
    df = df[~df.is_prefix_stopwords][["idiom"]]
    
    # add metadata fields
    df["id"] = df.apply(lambda row: secrets.token_hex(nbytes=10), axis=1)
    df["source"] = "MAGPIE"

    return df.to_dict('records')


def read_epic(path, min_length):
    idioms = []
    with open(path) as f:
        for idiom in f:
            idiom = _clean_idiom(idiom)
            if len(idiom.split(" ")) > min_length and not is_prefix_stopwords(idiom):
                idiom_dict = '{"idiom": "' + idiom + \
                             '", "id": "' + secrets.token_hex(nbytes=10) + \
                             '", "source": "EPIC"}'
                idioms.append(json.loads(idiom_dict))

    return idioms


def read_ef(path, min_length):
    idioms = []
    with open(path) as f:
        for idiom in f:
            idiom_str, source = idiom.split("\t")
            idiom_str = _clean_idiom(idiom_str)
            if len(idiom_str.split(" ")) > min_length and not is_prefix_stopwords(idiom):
                idiom_dict = '{"idiom": "' + idiom_str +\
                             '", "id": "' + secrets.token_hex(nbytes=10) +\
                             '", "source": "ep_'+ str(source[:-1]).replace(" ", "_") +\
                             '" }'
                idioms.append(json.loads(idiom_dict))
    return idioms


def read_lidiom(path, min_length, max_length):
    good_idioms = ['to be green with envy', 'to rub salt in the wound', 'to be at the end of one\'s rope','to place the cherry on top', 'to bite the dust', 'to have your cake and eat it, too', 'to paint the town red','to be green with envy', 'to be at the end of one\'s rope', 'to rub salt in the wound', 'to not be made of money']
    junk_idioms = ['asking someone what they are thinking about','an impassive facial expression hiding real feelings','in an extremely short time','asking someone what they are thinking about','when some task is very easy','doing things in a wrong manner','absolutely sure about something or someone','paying close attention and understanding the situation well','when everyone is facing the same challenges','asking someone what they are thinking about','you are trying to do something very difficult','would never like to do something','tricking someone as a joke','avoid something due to fear or uncertainty','make a lot of progress and improvement','something unpleasant that must be accepted or endured','receive information indirectly, similar to a rumor','a treacherous person, especially one who feigns friendship','gossip or accusations are often substantiated by fact','do something or say something exactly right','a life filled with excitement','someone who lacks intelligence','something good that isn\'t recognized at first','something that will never ever happen','getting ready to hard work','something that suddenly and unexpectedly occurs','unadorned facts, without concealment or embellishment','a person saves yourself of a danger situation','a person very kind','following the rules literally','someone who expresses an idea','nonsense or meaningless speech or writing','something or someone too old']
    junk = junk_idioms + ["http", "\u200b", ";", ":"]

    idioms = []
    g = Graph()
    g.parse(path, format="turtle")

    for triple in g:
        idiom = _clean_idiom(triple[2])
        is_junk = any(junk_word in idiom for junk_word in junk)
        is_junk = is_junk or idiom.startswith("to ") and idiom not in good_idioms
        if not is_junk and min_length < len(idiom.split(" ")) < max_length and not is_prefix_stopwords(idiom):
            idiom_dict = '{"idiom": "' + idiom + \
                         '", "id": "' + secrets.token_hex(nbytes=10) + \
                         '", "source": "LIdiom"}'
            idioms.append(json.loads(idiom_dict))
    return idioms


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--magpie_path",
                        default=r"data/MAGPIE_filtered_split_random.jsonl",
                        type=str,
                        required=False)

    parser.add_argument("--ef_path",
                        default=r"data/ef_idioms.txt",
                        type=str,
                        required=False)

    parser.add_argument("--epic_path",
                        default= r"data/EPIC_unique_Idioms.txt",
                        type=str,
                        required=False)

    parser.add_argument("--lidiom_path",
                        default=r"data/LIdioms_english.ttl",
                        type=str,
                        required=False)

    parser.add_argument("--out_path",
                        default=r"data/filtered_idioms.jsonl",
                        type=str,
                        required=False)

    parser.add_argument("--min_prompt_len",
                        default=3,
                        type=int,
                        required=False)

    parser.add_argument("--max_prompt_len",
                        default=9,
                        type=int,
                        required=False)

    parser.add_argument("--compute_stats_only",
                        default=False,
                        action='store_true',
                        required=False)

    return parser.parse_args()


def print_idioms_stats(idioms, source_str):
    print("Stats for :", source_str)
    print("mean:", np.array([len(idiom.split()) for idiom in idioms]).mean())
    print("std:", np.array([len(idiom.split()) for idiom in idioms]).std())
    random.shuffle(idioms)
    sample_idioms = idioms[:5]
    print(sample_idioms)


if __name__ == '__main__':
    args = parse_args()

    magpie_idioms = read_magpie(args.magpie_path, args.min_prompt_len)
    ef_idioms = read_ef(args.ef_path, args.min_prompt_len)
    epic_idioms = read_epic(args.epic_path, args.min_prompt_len)
    lidiom_idioms = read_lidiom(args.lidiom_path, args.min_prompt_len, args.max_prompt_len)
    all_idioms = magpie_idioms + ef_idioms + epic_idioms + lidiom_idioms
    unique_idioms = {v['idiom']: v for v in all_idioms}.values()


    if not args.compute_stats_only:
        with open(args.out_path, 'w') as outfile:
            for entry in unique_idioms:
                json.dump(entry, outfile)
                outfile.write('\n')

    print("{} idioms were collected overall, {} from magpie, "
          "{} from ef, {} from lidiom, {} from epic".format(len(unique_idioms), len(magpie_idioms), len(ef_idioms),
                                                            len(lidiom_idioms), len(epic_idioms)))

    print_idioms_stats([v['idiom'] for v in unique_idioms], "total")
    print_idioms_stats([v['idiom'] for v in magpie_idioms], "magpie")
    print_idioms_stats([v['idiom'] for v in ef_idioms], "ef")
    print_idioms_stats([v['idiom'] for v in lidiom_idioms], "lidiom")
    print_idioms_stats([v['idiom'] for v in epic_idioms], "epic")
