import json
import codecs
import os
import random
import sys
import math
import string
import re
from tqdm import tqdm
import argparse

ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)

#######################################

def find_position_for_spans(context, span):
    return context.find(span)

def find_sentence_for_spans(context_sentences, span):
    for sent in context_sentences:
        if span in sent:
            return sent
    return None

def compose_question_for_iteration(question, spans, sep_token='except', end_with_question_mark=False):
    if sep_token in ['except']:
        if question[-1] == '?':
            question = question[:-1]
        question += f" , {sep_token} "
        for i, span in enumerate(spans):
            if span[-1] in ['.',',','?','!',';',':', '(']:
                span = span[:-1].strip()
            question += f"{span}"
            if i != len(spans) - 1:
                question += ' , '
            elif end_with_question_mark:
                question += ' ?'
        return question
    elif sep_token == '<unused0>':
        if question[-1] == '?':
            question = question[:-1]
        question += f" {sep_token} "
        for i, span in enumerate(spans):
            if span[-1] in ['.',',','?','!',';',':', '(']:
                span = span[:-1].strip()
            question += f"{span}"
            if i != len(spans) - 1:
                question += ' , '
            elif end_with_question_mark:
                question += ' ?'
        return question
    elif sep_token == 'none':
        for i, span in enumerate(spans):
            if span[-1] in ['.',',','?','!',';',':', '(']:
                span = span[:-1].strip()
            question += f" {span} "
        if end_with_question_mark:
            question += ' ?'
        return question
    else:
        print("Unknown sep_token")

def filter_sentence(question, candidates, bert_embedder, topk=10):
    query_embedding = bert_embedder.encode(question, convert_to_tensor=True)
    candidate_embeddings = bert_embedder.encode(candidates, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, candidate_embeddings, top_k=topk)
    hits = hits[0]      #Get the hits for the first query
    ret = []
    for hit in hits:
        ret.append({
            'corpus_id': hit['corpus_id'],
            'text': candidates[hit['corpus_id']],
            'score': float(hit['score'])
        })
        # print(candidates[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="allspan",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sep-token",
        default="except",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--answer-order",
        default="appear-order",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save-path",
        default="allspan_squad-iterative",
        type=str,
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int
    )
    parser.add_argument(
        "--multiple-seed",
        type=str
    )
    parser.add_argument(
        "--end-with-question-mark",
        action='store_true'
    )
    parser.add_argument(
        "--no-first-iteration",
        action='store_true'
    )
    args = parser.parse_args()

    if args.answer_order == 'sent-sim':
        from sentence_transformers import SentenceTransformer, util
        bert_embedder = SentenceTransformer('../../models/all-mpnet-base-v2')

    for split in ['train', 'dev']:
        data = json_load(f"../data/multispanqa/squad_{split}.json")['data']
        
        new_data = {
            'version': f"{split} {args.mode}",
            'data': []
        }

        for item in tqdm(data):
            passage = item['paragraphs'][0]['context']
            if args.answer_order == 'sent-sim':
                from nltk.tokenize import sent_tokenize
                context_sentences = sent_tokenize(passage)
            new_item = {
                'paragraphs': [
                    {
                        'context': passage,
                        'qas': []
                    }
                ],
                'title': item['title']
            }
            for qa in item['paragraphs'][0]['qas']:
                if args.mode == 'multispan':
                    if len(qa['answers']) <= 1:
                        continue
                elif args.mode == 'allspan':
                    if len(qa['answers']) == 0:
                        continue
                answers = [(_['text'], _['answer_start']) for _ in qa['answers']]

                if len(answers) == 0:
                    continue

                if args.answer_order == 'appear-order':
                    answers = sorted(answers, key=lambda x: x[1])

                elif args.answer_order == 'random-order':
                    if args.seed != None:
                        random.seed(args.seed)
                    elif args.multiple_seed != None:
                        seed_list = [int(_) for _ in args.multiple_seed.split(",")]
                    random.shuffle(answers)

                elif args.answer_order == 'sent-sim':
                    pass
                    spans_and_sentences = [(span,find_sentence_for_spans(context_sentences, span)) for span in qas['answer']['spans']]
                    spans_and_sentences = [(_, s)for _, s in spans_and_sentences if s != None]
                    
                    if len(spans_and_sentences) == 0:
                        continue

                    if len(spans_and_sentences) > 1:
                        selected_sentences = filter_sentence(qa['question'], [s for _, s in spans_and_sentences], bert_embedder, topk=len(spans_and_sentences))
                        
                        answers_with_new_order = [spans_and_sentences[s['corpus_id']][0] for s in selected_sentences]
                    else:
                        answers_with_new_order = [spans_and_sentences[0][0]]
                    answers = [(span, find_position_for_spans(item['passage'], span)) for span in answers_with_new_order]
                    answers = [a for a in answers if a[1] != -1]

                new_qas_cnt = 0
                for i in range(len(answers)):
                    if i == 0:
                        if args.no_first_iteration:
                            continue
                        new_q = qa['question']
                        if new_q[-1] != ' ?' and args.end_with_question_mark:
                            new_q += ' ?'
                    else:
                        new_q = compose_question_for_iteration(qa['question'], [a[0] for a in answers[:i]], sep_token=args.sep_token, end_with_question_mark=args.end_with_question_mark)
                    new_item['paragraphs'][0]['qas'].append({
                        'id': f"{qa['id']}_{i}",
                        'question': new_q,
                        'answers': [{
                            'text': answers[i][0],
                            'answer_start': answers[i][1]
                        }],
                        "is_impossible": False
                    })

                new_q = compose_question_for_iteration(qa['question'], [a[0] for a in answers], sep_token=args.sep_token, end_with_question_mark=args.end_with_question_mark)
                new_item['paragraphs'][0]['qas'].append({
                    'id': f"{qa['id']}_{len(answers)}",
                    'question': new_q,
                    'answers': [],
                    "is_impossible": True
                })

            if len(new_item['paragraphs'][0]['qas']) == 0:
                continue
            new_data['data'].append(new_item)
        
        save_path = f"{args.save_path}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        json_dump(new_data, os.path.join(save_path, f"{split}.json"))
                    
                    
                    

'''
python3 convert_multispanqa_to_squad-iterative.py \
--mode allspan \
--sep-token except \
--answer-order  appear-order \
--save-path allspan_squad-style-iterative_except-appear-order

'''
