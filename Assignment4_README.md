# Assignment 4: Fine-tuning Language Models
**Due date:** Friday, Nov 14th at 12:59PM EST

**Academic Honesty:** Please see the course syllabus for information about collaboration in this course. While you may discuss the assignment with other students, all work you submit must be your own!

## Overview

In this assignment, you will train two types of language models, encoder-only model (BERT) and an encoder-decoder model. The main goal is learning how to fine-tune language models for a specific task, and understanding the challenges involved in it.

In Part 1, you will fine-tune an encoder-only model for sentiment classification dataset, focusing on transformation of data to understand generalization.  
In Part 2, you will train an encoder-decoder model to translate natural language instruction for flight booking into SQL query. Here, you will study conditional generation: generating a variable length, structured output (SQL query) given a variable length input.

## Starter Repo

You will require GPUs for this HW. Please first go through the file `README.md` to set up the environment required for the project.

For Part 1 of the homework, you were provided with `main.py` and `utils.py` as starter code to fine-tune an encoder-only model on a sentiment classification dataset.

For Part 2 of the homework, the data can be found under the `data/` directory.  
The database you will be evaluating queries on is `flight_database.db`, with the `flight_database.schema` file containing the database schema.

The text-to-SQL data is split into training, development and test sets. Files with the `.nl` extension contain natural language instructions, while `.sql` files contain corresponding SQL queries.

Starter code contains utility functions for SQL evaluation (`utils.py`).  
A training skeleton for T5 is provided in `train_t5.py`.

## Submissions

All submissions on Gradescope.

### Written (PDF)

Must include:

1. Answers to written questions  
2. **Link to your GitHub repository** (code for Part I and II)  
3. **Link to Google Drive** containing model checkpoint used for Q7  
4. Extra credit checkpoint link if applicable  

### Programming Outputs

- **Q1:** `out_original.txt`  
- **Q2:** `out_transformed.txt`  
- **Q3:** `out_augmented_original.txt` and `out_augmented_transformed.txt`  
- **Q7:** `t5_ft_experiment_test.pkl` and `t5_ft_experiment_test.sql`  
- **Extra Credit:** `t5_ft_experiment_ec_test.pkl` and `t5_ft_experiment_ec_test.sql`

# Part 1 — Fine-tuning BERT (50 pts)

We will explore training BERT for sentiment classification, focusing on OOD evaluation.

## Q1. Fine-tuning BERT model (10 pts coding)

Complete the missing parts in the training loop inside `do_train()` in `main.py`.

### Initial Testing
```bash
python3 main.py --train --eval --debug train
```
Expected: Accuracy **> 88%**

### Full Training
```bash
python3 main.py --train --eval
```
Expected: Accuracy **> 91%**

Submit: **`out_original.txt`**

## Q2. Data Transformations

Create reasonable transformations acting as out-of-distribution (OOD) evaluation.

Valid examples:  
- synonym replacement  
- typos  
Must preserve label meaning.

### Q2.1 (10 pts written)

Describe the transformation clearly so someone else can reimplement it.

### Q2.2 (15 pts coding)

Implement in `custom_transform()` in `utils.py`.

Debug:
```bash
python3 main.py --eval transformed --debug transformation
```

Evaluate:
```bash
python3 main.py --eval transformed
```

### Grading based on accuracy drop

- ≤ 4 point drop → **partial credit (8/15)**  
- > 4 point drop → **full credit (15/15)**

Submit: **`out_transformed.txt`**

## Q3. Data Augmentation

Augment training data with **5,000 transformed examples**.  
Complete function `create_augmented_dataloader()` in `main.py`.

Train:
```bash
python3 main.py --train augmented --eval transformed
```

### Evaluate

Original test data:
```bash
python3 main.py --eval --model_dir out_augmented
```

Transformed test data:
```bash
python3 main.py --eval transformed --model_dir out_augmented
```

### Q3 Written (15 pts)

Report:

- Accuracy on original and transformed test data  
- Whether augmentation improved transformed performance  
- Impact on original test accuracy  
- Explain observations intuitively  
- State one limitation of this augmentation method  

Submit:  
- **`out_augmented_original.txt`**  
- **`out_augmented_transformed.txt`**

# Part 2 — Fine-tuning T5 for Text-to-SQL (50 pts)

You will fine-tune **T5-small** (encoder-decoder) for predicting SQL queries from natural language.

Evaluation uses **execution-based metrics**:

- **Record F1** (main metric)  
- **Record Exact Match**  
- **SQL Query Exact Match**

Use `evaluate.py` to check your format.

## Q4. Data Statistics and Processing (5 pts written)

Fill two tables (before and after preprocessing):

- number of examples  
- mean natural language length  
- mean SQL length  
- vocabulary sizes  
- etc.

Use T5 tokenizer for statistics.

## Q5. T5 Fine-tuning Details (10 pts written)

Fill Table 3 describing:

- data processing steps  
- tokenization method  
- architecture choices (full fine-tune or partial)  
- hyperparameters (lr, batch size, stopping)  

Must be detailed enough for reproduction.

## Q6. Results (10 pts written)

### Quantitative

Fill Table 4:

- Dev results  
- Test results (must match leaderboard)  
- Include ablation variants if applicable  

### Qualitative

Fill Table 5:

- At least **3 types of errors**  
- Provide example snippet  
- Describe error  
- Provide statistics in **COUNT/TOTAL** form  
- If multiple models share error type, provide separate stats

## Q7. Test Performance Evaluation (25 pts coding)

Leaderboard metric: **Record F1**

To receive full credit: **F1 ≥ 65**  
Partial credit:
```
score = (your_f1 / 65) * 25
```

Submit:

- `t5_ft_experiment_test.pkl`
- `t5_ft_experiment_test.sql`

Excessive test submissions may result in deductions.

# Extra Credit

### Extra Credit 1  
Top 3 leaderboard → **+1% course grade**

### Extra Credit 2 — Train T5 From Scratch (**+1.5%**)  
- Start with random weights  
- May design your own SQL tokenizer  
- Must achieve **≥ 50 F1**  
- Must add new versions of Tables 2, 3, 4, 5  

Submit:

- `t5_ft_experiment_ec_test.pkl`
- `t5_ft_experiment_ec_test.sql`

# Acknowledgment

- Part I from He He (adapted from NL-Augmenter)  
- Part II adapted from LM-class.org (Omer Gul, Anne Wu, Yoav Artzi)
