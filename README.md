# How Many Answers Should I Give? An Empirical Study of Multi-Answer Reading Comprehension
Repo for *How Many Answers Should I Give? An Empirical Study of Multi-Answer Reading Comprehension*, Findings of ACL 2023

Arxiv preprint: TBA

## Annotation Results
The annotated data in our paper are in the `dataset.zip` file.
### Size

| Dataset | DROP | Quoref | MultiSpanQA | Total |
| --- | --- | --- | --- | --- |
| **Number of annotated instances** | 3,133 | 2,418 | 1,306 | 6,857 |

### Format

The format of each dataset is as follows (adapted from the format of SQuAD).

```json
[
    {
        "paragraphs": [
            {
                "context": "As of the census of 2000...",
                "qas": [
                    {
                        "id": "db314f06-4dc0-4cbd-a1d7-db195c51977c",
                        "question": "Which two ancestries made up the same percentage of the population?",
                        "answers": [
                            {
                                "text": "danish",
                                "answer_start": 723
                            },
                            {
                                "text": "Italian",
                                "answer_start": 746
                            }
                        ],
                        "is_impossible": false,
                        "annotated_type": "question-dependent_with-clue-words",
                        "clue_word": "two",
                        "clue_word_types": [
                            "cardinal"
                        ]
                    }
                ]
            }
        ]
    }
]
```

`"annotated_type"` has 4 options: "passage-dependent" , "question-dependent_without-clue-words" , "question-dependent_with-clue-words" , "bad-annotation".

For with-clue-word questions, `clue_word` indicate the clue word, and `clue_word_types` have 5 types: "alternative" , "cardinal" , "comparative/superlative" , "ordinal" , "other_lexical_meaning".

## Code

### Iterative 

Our implementation is based on the scripts of MRC implemented by *Huggingface*. `convert_multispanqa_to_squad-iterative.py` converts datasets to iterative format. `run_squad.py` is for fine-tuning and `iterative_inference.py` to iteratively inference.

Requires:

+ transformers==3.5.1
