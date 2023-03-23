from symphony_net.src.preprocess import preprocess_midi
from symphony_net.src.preprocess import get_bpe_data
from symphony_net.src.fairseq import make_data
from symphony_net.src import encoding
from symphony_net.src.fairseq.gen_utils import process_prime_midi, gen_one, get_trk_ins_map, get_note_seq, note_seq_to_midi_file, music_dict
import os, traceback, time, warnings, sys, random
from tqdm import tqdm
from p_tqdm import p_uimap
from functools import partial
from collections import Counter
import subprocess#, multiprocessing
import json
from pprint import pprint

class Tokenizer:
    def __init__(self, ratio = 4, merge_cnt= 700, char_cnt= 128, workers= 32):
        self.main_path = "decoder/symphony_net/"
        self.ratio = ratio
        self.merge_cnt = merge_cnt
        self.char_cnt = char_cnt
        self.workers = workers
    
    def __preprocess_midi(self):
        warnings.filterwarnings('ignore')
        midis_path = self.main_path + "data/midis"
        file_paths = []
        for path, directories, files in os.walk(midis_path):
            for file in files:
                if file.endswith(".mid") or file.endswith(".MID"):
                    file_path = path + "/" + file
                    file_paths.append(file_path)

        # run multi-processing midi extractor
        preprocess_midi.mp_handler(file_paths)
    
    def __byte_pair_encode(self):
        start_time = time.time()

        raw_corpus_path = self.main_path + "data/preprocessed/raw_corpus.txt"
        bpe_raw_corpus_path = self.main_path + "data/preprocessed/raw_corpus_bpe.txt"
        output_dir = self.main_path + "data/bpe_res/"

        paragraphs = []

        os.makedirs(output_dir, exist_ok=True)
        raw_data = []
        with open(raw_corpus_path, 'r') as f:
            for line in tqdm(f, desc="reading original txt file..."):
                raw_data.append(line.strip())

        chord_dict = Counter()
        before_total_tokens = 0
        for sub_chord_dict, l_toks in p_uimap(get_bpe_data.count_single_mulpies, raw_data, num_cpus=self.workers):
            chord_dict += sub_chord_dict
            before_total_tokens += l_toks
        
        mulpi_list = sorted(chord_dict.most_common(), key=lambda x: (-x[1], x[0]))
        with open(output_dir+'ori_voc_cnt.txt', 'w') as f:
            f.write(str(len(mulpi_list)) + '\n')
            for k, v in mulpi_list:
                f.write(''.join(k) + ' ' + str(v) + '\n')
        with open(output_dir+'codes.txt', 'w') as stdout:
            with open(output_dir+'merged_voc_list.txt', 'w') as stderr:
                subprocess.run(['pwd'])
                subprocess.run(['./' + self.main_path + '/music_bpe_exec', 'learnbpe', f'{self.merge_cnt}', output_dir+'ori_voc_cnt.txt'], stdout=stdout, stderr=stderr)
        print(f'learnBPE finished, time elapsed:　{time.time() - start_time}')
        start_time = time.time()

        merges, merged_vocs = get_bpe_data.load_before_apply_bpe(output_dir)
        divide_res, divided_bpe_total, bpe_freq = get_bpe_data.apply_bpe_for_word_dict(mulpi_list, merges, merged_vocs)
        with open(output_dir+'divide_res.json', 'w') as f:
            json.dump({' '.join(k):v for k, v in divide_res.items()}, f)
        with open(output_dir+'bpe_voc_cnt.txt', 'w') as f:
            for voc, cnt in bpe_freq.most_common():
                f.write(voc + ' ' + str(cnt) + '\n')
        ave_len_bpe = sum(k*v for k, v in divided_bpe_total.items()) / sum(divided_bpe_total.values())
        ave_len_ori = sum(len(k)*v for k, v in mulpi_list) / sum(v for k, v in mulpi_list)
        print(f'average mulpi length original:　{ave_len_ori}, average mulpi length after bpe: {ave_len_bpe}')
        print(f'applyBPE for word finished, time elapsed:　{time.time() - start_time}')
        start_time = time.time()

        # applyBPE for corpus

        after_total_tokens = 0
        with open(bpe_raw_corpus_path, 'w') as f:
            for x in tqdm(raw_data, desc="writing bpe data"): # unable to parallelize for out of memory
                new_toks = get_bpe_data.apply_bpe_for_sentence(x, merges, merged_vocs, divide_res)
                after_total_tokens += len(new_toks) // self.ratio
                f.write(' '.join(new_toks) + '\n')
        print(f'applyBPE for corpus finished, time elapsed:　{time.time() - start_time}')
        print(f'before tokens: {before_total_tokens}, after tokens: {after_total_tokens}, delta: {(before_total_tokens - after_total_tokens) / before_total_tokens}')

    def __make_data(self):
        config_path = self.main_path + "config.sh"

        # --------- slice multi-track ----
        PAD = 1
        EOS = 2
        BOS = 0

        SEED, SAMPLE_LEN_MAX, totpiece, RATIO, bpe, map_meta_to_pad = None, None, None, None, None, None
        print('config.sh: ')
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    break
                print(line)
                line = line.split('=')
                assert len(line) == 2, f'invalid config {line}'
                if line[0] == 'SEED':
                    SEED = int(line[1])
                    random.seed(SEED)
                elif line[0] == 'MAX_POS_LEN':
                    SAMPLE_LEN_MAX = int(line[1])
                elif line[0] == 'MAXPIECES':
                    totpiece = int(line[1])
                elif line[0] == 'RATIO':
                    RATIO = int(line[1])
                elif line[0] == 'BPE':
                    bpe = int(line[1])
                elif line[0] == 'IGNORE_META_LOSS':
                    map_meta_to_pad = int(line[1])

        assert SEED is not None, "missing arg: SEED"
        assert SAMPLE_LEN_MAX is not None, "missing arg: MAX_POS_LEN"
        assert totpiece is not None, "missing arg: MAXPIECES"
        assert RATIO is not None, "missing arg: RATIO"
        assert bpe is not None, "missing arg: BPE"
        assert map_meta_to_pad is not None, "missing arg: IGNORE_META_LOSS"

        bpe = "" if bpe == 0 else "_bpe"
        raw_corpus = f'raw_corpus{bpe}'
        model_name = f"linear_{SAMPLE_LEN_MAX}_chord{bpe}"
        raw_data_path = f'{self.main_path}data/preprocessed/{raw_corpus}.txt'
        output_dir = f'{self.main_path}data/model_spec/{model_name}_hardloss{map_meta_to_pad}/'
        
        start_time = time.time()
        raw_data = []
        with open(raw_data_path, 'r') as f:
            for line in tqdm(f, desc='reading...'):
                raw_data.append(line.strip())
                if len(raw_data) >= totpiece:
                    break

        sub_vocabs = dict()
        for i in range(RATIO):
            sub_vocabs[i] = set()

        for ret_sets in p_uimap(partial(make_data.makevocabs, ratio=RATIO), raw_data, num_cpus=self.workers, desc='setting up vocabs'):
            for i in range(RATIO):
                sub_vocabs[i] |= ret_sets[i]

        voc_to_int = dict()
        for type in range(RATIO):
            sub_vocabs[type] |= set(('<bos>', '<pad>', '<eos>', '<unk>'))
            sub_vocabs[type] -= set(('RZ', 'TZ', 'YZ'))
            sub_vocabs[type] = sorted(list(sub_vocabs[type]), key=encoding.sort_tok_str)
            voc_to_int.update({v:i for i,v in enumerate(sub_vocabs[type]) }) 
        output_dict = sorted(list(set(voc_to_int.values())))
        max_voc_size = max(output_dict)
        print("max voc idx: ", max_voc_size)

        os.makedirs(output_dir + 'bin/', exist_ok=True)
        with open(output_dir + 'bin/dict.txt', 'w') as f:
            for i in range(4, max_voc_size+1): # [4, max_voc_size]
                f.write("%d 0\n"%i)

            
        os.makedirs(output_dir + 'vocabs/', exist_ok=True)
        for type in range(RATIO):
            sub_vocab = sub_vocabs[type]
            with open(output_dir + 'vocabs/vocab_%d.json'%type, 'w') as f:
                json.dump({i:v for i,v in enumerate(sub_vocab)}, f)
        with open(output_dir + 'vocabs/ori_dict.json', 'w') as f:
            json.dump(voc_to_int, f)
        print('sub vocab size:', end = ' ')
        for type in range(RATIO):
            print(len(sub_vocabs[type]), end = ' ')
        print()
        with open(f'{self.main_path}/vocab.sh', 'w') as f:
            for type in range(RATIO):
                f.write(f'SIZE_{type}={len(sub_vocabs[type])}\n')

        totpiece = len(raw_data)
        print("total pieces: {:d}, create dict time: {:.2f} s".format(totpiece, time.time() - start_time))

        raw_data = make_data.myshuffle(raw_data)
        os.makedirs(output_dir + 'bin/', exist_ok=True)
        train_size = min(int(totpiece*0.99), totpiece-2)
        splits = {'train': raw_data[:train_size], 'valid': raw_data[train_size:-1], 'test': raw_data[-1:]}
        print("ratio: ",RATIO)

        voc_to_int.update({x:(PAD if map_meta_to_pad == 1 else BOS) for x in ('RZ', 'TZ', 'YZ')}) 
        for mode in splits:
            print(mode)
            make_data.mp_handler(self.main_path, splits[mode], voc_to_int, output_dir + f'bin/{mode}', ratio=RATIO, sample_len_max=SAMPLE_LEN_MAX)
    
    def vocab_generate(self):
        self.__preprocess_midi()
        self.__byte_pair_encode()
        self.__make_data()

    def encode(
            self,
            midi_file_path,
            max_measure_cnt = 5,
            max_chord_measure_cnt = 0
        ):
        MAX_POS_LEN = 4096
        PI_LEVEL = 2
        IGNORE_META_LOSS = 1
        RATIO = 4
        BPE = "_bpe" # or ""

        DATA_BIN=f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
        CHECKPOINT_SUFFIX=f"{DATA_BIN}_PI{PI_LEVEL}"
        DATA_BIN_DIR = f"{self.main_path}data/model_spec/{DATA_BIN}/bin/"
        DATA_VOC_DIR = f"{self.main_path}data/model_spec/{DATA_BIN}/vocabs/"
        BPE_RES_PATH = f"{self.main_path}data/bpe_res/"
        music_dict.load_vocabs_bpe(DATA_VOC_DIR, BPE_RES_PATH if BPE == '_bpe' else None)
        prime, ins_label = process_prime_midi(midi_file_path, max_measure_cnt, max_chord_measure_cnt)
        return prime, ins_label


tokenizer = Tokenizer()
tokenizer.vocab_generate();
# prime, ins_label = tokenizer.encode(
#     "/home/tnguy231/VIVY/VIVYNet/Decoder/symphony_net/data/midis/ty_maerz_format0.mid"
# )
# pprint(prime)