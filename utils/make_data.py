# Import
import numpy as np
from p_tqdm import p_uimap
import random, json, time, os
from tqdm import tqdm
import torch
import sys, os, multiprocessing
from collections import Counter
from pprint import pprint
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder
from functools import partial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .encoding import sort_tok_str
import subprocess

PAD = 1
EOS = 2
BOS = 0

WORKERS = 32


def process_single_piece(bundle_input, ratio, sample_len_max):
    line, str2int, index_num = bundle_input

    if isinstance(line, str):
        str_toks = line.split()
    else:
        str_toks = line

    measures = []
    cur_mea = []
    max_rel_pos = 0
    mea_tok_lengths = []

    rel_pos = 0

    for idx in range(0, len(str_toks), ratio):
        c = str_toks[idx][0]

        if c.lower() == "m":  # BOM Token
            if len(cur_mea) > 0:  # exlude first bom
                measures.append(cur_mea)
                mea_tok_lengths.append(len(cur_mea) // (ratio + 1))
            cur_mea = []
            if rel_pos > max_rel_pos:
                max_rel_pos = rel_pos
            rel_pos = 0
        elif c.lower() == "h":  # chord token
            if rel_pos > max_rel_pos:
                max_rel_pos = rel_pos
            rel_pos = 0
        elif c.lower() == "n":  # CC/NT Token
            if rel_pos > max_rel_pos:
                max_rel_pos = rel_pos
            rel_pos = 1
        elif c.lower() == "p":  # 　pos token
            rel_pos += 2
        else:  # on token
            pass
        # for i in str_toks[idx:idx+ratio]:
        #     print(str2int[i])
        #     print(idx, ratio, i)
        #     print("\n")
        cur_mea += [str2int[x] for x in str_toks[idx : idx + ratio]] + [
            rel_pos - 1 if c.lower() == "p" else rel_pos
        ]

    # TODO: how to design rel_pos and measure pos?
    if len(cur_mea) > 0:
        measures.append(cur_mea)
        mea_tok_lengths.append(len(cur_mea) // (ratio + 1))
        if rel_pos > max_rel_pos:
            max_rel_pos = rel_pos

    # tmp = get_mea_cnt(str_toks, ratio)
    # if get_mea_cnt(str_toks, ratio) != len(measures):
    #     print(f'{tmp} {len(measures)} {len(mea_tok_lengths)}')

    len_cnter = Counter()
    for l in mea_tok_lengths:
        len_cnter[l // 10] += 1

    for idx in range(1, len(mea_tok_lengths)):
        mea_tok_lengths[idx] += mea_tok_lengths[idx - 1]

    def get_cur_tokens(s, t):  # return total cnt of tokens in measure [s, t]
        return mea_tok_lengths[t] - (mea_tok_lengths[s - 1] if s > 0 else 0)

    maxl = 1
    for s in range(len(mea_tok_lengths)):
        t = s + maxl - 1

        while (
            t < len(mea_tok_lengths) and get_cur_tokens(s, t) < sample_len_max
        ):
            t += 1

        t = min(t, len(mea_tok_lengths) - 1)
        maxl = max(maxl, t - s + 1)

    return measures, len_cnter, max_rel_pos, maxl, index_num


def mp_handler(
    raw_data, str2int, output_file, ratio, sample_len_max, num_workers=WORKERS
):
    begin_time = time.time()

    merged_sentences = [None] * len(raw_data)
    mea_cnt_dis = Counter()
    mea_len_dis = Counter()
    max_rel_pos = 0
    maxl = 0
    # print('with multiprocessing.Pool(num_workers) as p')
    # with multiprocessing.Pool(num_workers) as p:
    #     for sentences, len_cnter, pos, l, index_num in \
    #         p.imap_unordered(
    #             partial(process_single_piece,
    #                 ratio=ratio,
    #                 sample_len_max=sample_len_max),
    #             [(x, str2int, idx) for idx, x in enumerate(raw_data)]
    #         ):
    #         merged_sentences[index_num] = sentences
    #         mea_len_dis += len_cnter
    #         max_rel_pos = max(max_rel_pos, pos)
    #         maxl = max(maxl, l)

    for idx, x in tqdm(enumerate(raw_data), desc="processing..."):
        sentences, len_cnter, pos, l, index_num = process_single_piece(
            (x, str2int, idx), ratio=ratio, sample_len_max=sample_len_max
        )
        merged_sentences[index_num] = sentences
        mea_len_dis += len_cnter
        max_rel_pos = max(max_rel_pos, pos)
        maxl = max(maxl, l)

    # input('input')
    print(
        f"measure collection finished, total " +
        f"{sum(len(x) for x in merged_sentences)} measures, time elapsed: " +
        f"{time.time()-begin_time} s"
    )
    print(f"max cnt in a sample (rel_pos, measure): {max_rel_pos}, {maxl}")

    begin_time = time.time()

    if "train" in output_file:
        with open("../data/temp/vocab.sh", "a") as f:
            f.write(f"MAX_REL_POS={max_rel_pos+5}\n")
            f.write(f"MAX_MEA_POS={maxl*3+5}\n")
        with open("../data/temp/mea_cnt_dis.txt", "w") as f:
            for k, v in sorted(mea_cnt_dis.items()):
                f.write(f"{k*10} {v}\n")
        with open("../data/temp/mea_len_dis.txt", "w") as f:
            for k, v in sorted(mea_len_dis.items()):
                f.write(f"{k*10} {v}\n")
    print("merged_sentences:\t", len(merged_sentences))
    count = 0
    ds = MMapIndexedDatasetBuilder(output_file + ".bin", dtype=np.uint16)
    for doc in tqdm(merged_sentences, desc="writing bin file"):
        for sentence in doc:
            ds.add_item(torch.IntTensor(sentence))
        ds.add_item(torch.IntTensor([EOS]))
        count += len(doc)
    ds.finalize(output_file + ".idx")
    print("total count:", count)
    print(
        "write binary finished, write time elapsed {:.2f} s".format(
            time.time() - begin_time
        )
    )


def makevocabs(line, ratio):
    toks = line.split()
    ret_sets = []
    for i in range(ratio):
        sub_toks = toks[i::ratio]
        ret_sets.append(set(sub_toks))
    return ret_sets


def midi_binarize(train_ratio: float) -> None:
    """MIDI Binarization Method

    Description:
        Binarizes the MIDI tokens into .idx and .bin files

    Information:
        :param train_ratio: the ration between training and validation data
        :type train_ratio: float
        :return: None
        :rtype: None

    """
    # Variable declaration
    SEED = None
    SAMPLE_LEN_MAX = None
    totpiece = None
    RATIO = None
    bpe = None
    map_meta_to_pad = None

    # Read the config
    with open("./config.sh", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                break

            line = line.split("=")
            assert len(line) == 2, f"invalid config {line}"
            if line[0] == "SEED":
                SEED = int(line[1])
                random.seed(SEED)
            elif line[0] == "MAX_POS_LEN":
                SAMPLE_LEN_MAX = int(line[1])
            elif line[0] == "MAXPIECES":
                totpiece = int(line[1])
            elif line[0] == "RATIO":
                RATIO = int(line[1])
            elif line[0] == "BPE":
                bpe = int(line[1])
            elif line[0] == "IGNORE_META_LOSS":
                map_meta_to_pad = int(line[1])

    # Asserts
    assert SEED is not None, "missing arg: SEED"
    assert SAMPLE_LEN_MAX is not None, "missing arg: MAX_POS_LEN"
    assert totpiece is not None, "missing arg: MAXPIECES"
    assert RATIO is not None, "missing arg: RATIO"
    assert bpe is not None, "missing arg: BPE"
    assert map_meta_to_pad is not None, "missing arg: IGNORE_META_LOSS"

    # Set the values for the parameters
    bpe = "" if bpe == 0 else "_bpe"
    raw_corpus = f"data.y"
    raw_data_path = f"../data/tokens/{raw_corpus}"
    output_dir = f"../data/final/labels/"

    # Start the timer clock and create a raw_data list
    start_time = time.time()
    raw_data = []

    # Send each line into the raw_data list
    with open(raw_data_path, "r") as f:
        for line in tqdm(f, desc="reading..."):
            raw_data.append(line.strip())
            # if len(raw_data) >= totpiece:
            #     break
            # print(line)
    print("raw length:", len(raw_data))

    # Create a component vocab list
    sub_vocabs = dict()
    for i in range(RATIO):
        sub_vocabs[i] = set()
    # Create the vocabs from the MIDI files
    for ret_sets in p_uimap(
        partial(makevocabs, ratio=RATIO),
        raw_data,
        num_cpus=32,
        desc="setting up vocabs",
    ):
        for i in range(RATIO):
            sub_vocabs[i] |= ret_sets[i]

    # Create the integers from the vocabularies
    voc_to_int = dict()
    for type in range(RATIO):
        sub_vocabs[type] |= set(("<bos>", "<pad>", "<eos>", "<unk>"))
        sub_vocabs[type] -= set(("RZ", "TZ", "YZ"))
        sub_vocabs[type] = sorted(list(sub_vocabs[type]), key=sort_tok_str)
        voc_to_int.update({v: i for i, v in enumerate(sub_vocabs[type])})
    output_dict = sorted(list(set(voc_to_int.values())))
    max_voc_size = max(output_dict)
    print("max voc idx: ", max_voc_size)

    # create bin folder and place the dictionary with the corresponding
    # contenting into it
    os.makedirs(output_dir + "bin/", exist_ok=True)
    with open(output_dir + "bin/dict.txt", "w") as f:
        for i in range(4, max_voc_size + 1):  # [4, max_voc_size]
            f.write("%d 0\n" % i)

    # Create a vocabs directory and necessary files
    # and place the vocab information there
    os.makedirs(output_dir + "vocabs/", exist_ok=True)
    for type in range(RATIO):
        sub_vocab = sub_vocabs[type]
        with open(output_dir + "vocabs/vocab_%d.json" % type, "w") as f:
            json.dump({i: v for i, v in enumerate(sub_vocab)}, f)
    with open(output_dir + "vocabs/ori_dict.json", "w") as f:
        json.dump(voc_to_int, f)
    print("sub vocab size:", end=" ")
    for type in range(RATIO):
        print(len(sub_vocabs[type]), end=" ")
    print()
    with open(f"../data/temp/vocab.sh", "w") as f:
        for type in range(RATIO):
            f.write(f"SIZE_{type}={len(sub_vocabs[type])}\n")

    # Print the total number of pieces
    totpiece = len(raw_data)
    print(
        "total pieces: {:d}, create dict time: {:.2f} s".format(
            totpiece, time.time() - start_time
        )
    )

    # Generate the split information
    os.makedirs(output_dir, exist_ok=True)
    train_size = int(totpiece * train_ratio)
    splits = {"train": raw_data[:train_size], "valid": raw_data[train_size:]}
    # print(">>>>>>>>>>>>>")
    # print(len(splits['train']))
    # print(len(splits['valid']))
    # print(">>>>>>>>>>>>>")

    # Update the vocabularies to int dictionary
    voc_to_int.update(
        {x: (PAD if map_meta_to_pad == 1 else BOS) for x in ("RZ", "TZ", "YZ")}
    )

    # Binzarize data
    for mode in splits:
        print(mode)
        mp_handler(
            splits[mode],
            voc_to_int,
            output_dir + f"bin/{mode}",
            ratio=RATIO,
            sample_len_max=SAMPLE_LEN_MAX,
        )
