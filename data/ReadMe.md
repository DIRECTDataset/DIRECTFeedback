# DIRECT-Feedback (data/feedback_data_partial.csv)

#### Descriptions

This dataset holds feedback presented in the context of interactive reading comprehension exercises. It is an extension of DIRECT based on RACE. For DIRECT, incorrect answers were constructed by selecting faulty answer options from a pre-defined multiple choice form. Those are often totally unrelated to the actual reading passage. We decided to construct additional data with more natural answers, including mistakes that students are likely to make in an environment where only the reading passage and no answer options are provided. For each question in the DIRECT dataset, one annotator in the student role constructs such an answer, then another annotator in the tutor role constructs the corresponding feedback. Both annotators are presented with the reading passage, the question, its correct answer, and the corresponding key sentences. Five annotators with some level of English proficiency worked on the student role, while two native English-speaking annotators worked on the tutor role. The latter were also asked to periodically review randomly selected portions of the constructed data, including both incorrect answers and tutor feedback (constructed by the other worker). They ensured that the percentage of erroneous data items remained below 5%. We call this new dataset DIRECT-Feedback.

#### Data Statistics

We maintain the split of the RACE data set which assigns each reading passage with all its questions to exactly one split.

| | Train | Validation | Test | | Total |
| :---: | :---: | :---: | :---: | :-: | :---: |
| **DIRECT**  |   |   |   | |   |
| Dialogues | 2707 (5066) | 301 | 301 || 5668  |
| Feedback Turns | 5026 (9431) | 475 | 525 || 10431  |
| **DIRECT-Feedback** |   |   |   | |   |
| Dialogues | 2722 (5095) | 302 | 307 || 5704 |
| Feedback Turns | 11440 (21463) | 1239 | 1280 || 23982 |
| **Total** |   |   |   | |   |
| Dialogues | 2722 (5095)  | 302  | 307  | | 5704  |
| Feedback Turns | 16466 (30894)  | 1714  | 1805  | | 34413  |

We publish the whole validation and test set. The training set is published in parts (around 50% of the original data).

#### Format
Data is provided as a single CSV file that holds one data item per line:
```
set\tfile_id\tquestion_id\tquestion\tkey_sentence\tcorrect_answer\twrong_answer\tfeedback
```
The first column indicates whether the feedback is part of the original DIRECT dataset or our augmentation DIRECT-Feedback.  
The second column provides a unique identifer that can be used to retrieve a reading passage using the provided JSON file `article-id_mapping.json`.  
The third and fourth column hold information on the question the student was asked to answer.  
The fifth column holds a subset of the reading passage's sentences that are relevant to the question.  
The sixth column holds the expected/correct answer to the question.  
The seventh column holds the incorrect answer provided by the student.  
The eigth column holds feedback in the form of hints, explanations or corrections as it was provided by the tutor role.  

#### Disclaimer
This repository holds information from the [RACE](https://www.cs.cmu.edu/~glai1/data/race/#:~:text=notes) dataset that is available under a non-profit research-only licence.
