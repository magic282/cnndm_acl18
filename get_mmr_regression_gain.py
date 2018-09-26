from find_oracle import load_data
from ast import literal_eval as make_tuple
import random
import math

from PyRouge.Rouge.Rouge import Rouge
from Document import Document

rouge = Rouge(use_ngram_buf=True)


def load_upperbound(filepath):
    res = []
    with open(filepath, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            sp = line.split('\t')
            if 'None' in sp[0]:
                comb = None
            else:
                comb = make_tuple(sp[0])
            score = float(sp[1])
            res.append((comb, score))
    return res


def get_mmr_order(oracle, doc):
    scores = [(rouge.compute_rouge([doc.summary_sents], [[doc.doc_sents[idx]]])['rouge-2']['f'][0]) for idx in oracle[0]]
    comb = zip(oracle[0], scores)
    comb = sorted(comb, key=lambda x: -x[1])
    selected = []
    left = [x[0] for x in comb[1:]]
    selected.append(comb[0][0])
    while len(left) > 0:
        candidates = [(selected + [x]) for x in left]
        scores = [(rouge.compute_rouge([doc.summary_sents], [[doc.doc_sents[idx] for idx in can]])['rouge-2']['f'][0])
                  for can in
                  candidates]
        tmp = zip(list(range(len(candidates))), scores)
        sorted_tmp = sorted(tmp, key=lambda x: -x[1])
        best_sent = left[sorted_tmp[0][0]]
        best_score = sorted_tmp[0][1]
        selected.append(best_sent)
        del left[sorted_tmp[0][0]]
    mmr_comb = tuple(selected)
    return mmr_comb


def get_mmr_regression(oracle, doc):
    selected = []
    selected_id = []
    prev_rouge = 0
    res_buf = []
    for sent_id in oracle:
        candidates = [(selected + [x]) for x in doc.doc_sents]
        cur_rouge = [(rouge.compute_rouge([doc.summary_sents], [can])['rouge-2']['f'][0]) for can in candidates]
        selected.append(doc.doc_sents[sent_id])
        selected_id.append(sent_id)
        out_rouge = [(x - prev_rouge) for x in cur_rouge]
        out_string = ' '.join([str(x) for x in out_rouge])
        res_buf.append(out_string)
        prev_rouge = max(cur_rouge)

    return tuple(selected_id), '\t'.join(res_buf)


def main(src_file, tgt_file, oracle_file, output_file):
    docs = load_data(src_file, tgt_file)
    oracles = load_upperbound(oracle_file)

    acc = 0
    count = 0
    for item in oracles:
        if item[0] is not None:
            acc += item[1]
            count += 1
    print('upper bound: {0}'.format(acc / count))

    count = 0
    with open(output_file, 'w', encoding='utf-8') as writer:
        for doc, oracle in zip(docs, oracles):
            count += 1
            if count % 100 == 0:
                print(count)
                rouge.ngram_buf = {}
            if oracle[0] is None:
                writer.write('None\t0' + '\n')
                continue
            oracle_with_order = get_mmr_order(oracle, doc)
            oracle_with_order, rouge_scores = get_mmr_regression(oracle_with_order, doc)
            writer.write('{0}\t{1}'.format(oracle_with_order, rouge_scores) + '\n')


if __name__ == "__main__":
    src_file = r"sample_data/train.txt.src.100"
    tgt_file = r"sample_data/train.txt.tgt.100"
    oracle_file = r"sample_data/train.rouge_bigram_F1.oracle.100"
    output_file = r"sample_data/train.rouge_bigram_F1.oracle.100.regGain"

    main(src_file, tgt_file, oracle_file, output_file)
