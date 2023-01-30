MAX_POS_LEN = 4096
PI_LEVEL = 2
IGNORE_META_LOSS = 1
RATIO = 4
BPE = "_bpe" # or ""

DATA_BIN=f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
CHECKPOINT_SUFFIX=f"{DATA_BIN}_PI{PI_LEVEL}"
DATA_BIN_DIR=f"/home/tnguy231/VIVY/VIVYNet/Decoder/symphony_net/data/model_spec/{DATA_BIN}/bin/"
DATA_VOC_DIR=f"/home/tnguy231/VIVY/VIVYNet/Decoder/symphony_net/data/model_spec/{DATA_BIN}/vocabs/"

from src.fairseq.gen_utils import process_prime_midi, gen_one, get_trk_ins_map, get_note_seq, note_seq_to_midi_file, music_dict
music_dict.load_vocabs_bpe(DATA_VOC_DIR, '/home/tnguy231/VIVY/VIVYNet/Decoder/symphony_net/data/bpe_res/' if BPE == '_bpe' else None)



midi_name = '/home/tnguy231/VIVY/VIVYNet/Decoder/symphony_net/test.mid'
test_midi_name = '/home/tnguy231/VIVY/VIVYNet/Decoder/symphony_net/symphony_6_2_(c)lucarelli.mid'
max_measure_cnt = 5
max_chord_measure_cnt = 0
prime, ins_label = process_prime_midi(midi_name, max_measure_cnt, max_chord_measure_cnt)
# prime1, ins_label1 = process_prime_midi(test_midi_name, max_measure_cnt, max_chord_measure_cnt)