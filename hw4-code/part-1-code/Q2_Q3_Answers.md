# Assignment 4 - Part 1: Written Answers

## Q2.1: Transformation Design (10 points)

### Transformation Name
**Hybrid Synonym Replacement and Typo Introduction**

### Detailed Description

This transformation applies two realistic text modifications that can occur in real-world user input:

#### Component 1: Synonym Replacement (30% probability per word)

**Process:**
1. Tokenize the input text using NLTK's `word_tokenize()` function to split the text into individual words and punctuation tokens.
2. For each alphabetic word (length ≥ 3 characters):
   - With 30% probability, attempt to replace it with a synonym.
   - Query WordNet for all synsets of the word (using the lowercased version).
   - Collect all lemma synonyms from all synsets.
   - Filter out:
     - The original word itself
     - Multi-word phrases (keep only single-word synonyms)
   - If synonyms exist, randomly select one from the filtered list.
   - Preserve original capitalization (if the original word was capitalized, capitalize the synonym).
   - Replace the word with the chosen synonym.
3. Skip words that:
   - Are not purely alphabetic (contain punctuation or numbers)
   - Are shorter than 3 characters
   - Have no synonyms available in WordNet

**Example:** "Titanic is the best movie I have ever seen." → "Titanic is the best film I have ever seen." (if "movie" → "film" replacement occurs)

#### Component 2: Typo Introduction (20% probability per word, if synonym replacement didn't occur)

**Process:**
1. For words not replaced by synonyms:
   - With 20% probability, introduce a single-character typo.
   - Only apply to words longer than 2 characters.
2. Typo generation:
   - Select a random position within the word (excluding first and last characters to keep the word recognizable).
   - Replace the character at that position with a nearby QWERTY keyboard key.
   - QWERTY neighbor mapping includes:
     - Vowels: 'a' → ['s', 'q', 'w'], 'e' → ['w', 'r', 'd'], 'i' → ['u', 'o', 'k'], etc.
     - Consonants: 't' → ['r', 'y', 'g'], 'n' → ['b', 'h', 'j', 'm'], etc.
   - Preserve original capitalization.
3. Skip words that:
   - Were already replaced by synonyms
   - Are too short (< 3 characters)
   - Don't have QWERTY neighbors for the selected character

**Example:** "movie" → "mocie" (if 'v' → 'c' typo occurs at position 2)

#### Text Reconstruction

After processing all tokens:
- Use `TreebankWordDetokenizer()` to reconstruct the text from tokens.
- Preserve punctuation and spacing.
- Maintain sentence structure.

**Complete Example:**
- Original: "Titanic is the best movie I have ever seen."
- Transformed: "Titanic is the best film I have ever seen." (synonym replacement)
- Or: "Titanic is the best mocie I have ever seen." (typo introduction)

### Why This Transformation Is Reasonable

1. **Synonym usage is natural**: People naturally use different words to express the same meaning (e.g., "movie" vs "film", "good" vs "great").
2. **Typos are realistic**: Real-world text often contains typos from mobile keyboards, fast typing, or autocorrect errors.
3. **Preserves meaning**: Both transformations maintain semantic meaning and sentiment labels, which is crucial for sentiment analysis.
4. **Realistic test scenarios**: These are transformations that could genuinely appear in test data from users.
5. **Balanced modification**: The probabilities (30% synonym, 20% typo) ensure most text remains recognizable while introducing meaningful variation.

### Implementation Details for Replication

**Required libraries:**
- `nltk` (for `word_tokenize`, `wordnet`, `TreebankWordDetokenizer`)

**Key parameters:**
- Synonym replacement probability: 0.3 (30%)
- Typo introduction probability: 0.2 (20%)
- Minimum word length: 3 characters
- Typo position: Random between positions 1 and len(word)-2 (exclusive of first and last)

**QWERTY keyboard mapping:**
- Map each letter to its physically adjacent keys on a QWERTY keyboard
- Use this mapping to select replacement characters for typos

**WordNet usage:**
- Query `wordnet.synsets(word)` for all synsets
- Extract synonyms via `syn.lemmas()` for each synset
- Filter to single-word synonyms that differ from the original

---

## Q3: Data Augmentation Analysis (15 points)

### Accuracy Values

**Augmented Model Performance:**
- Original test data: **92.764%**
- Transformed test data: **89.116%**

**Baseline Model Performance (for comparison):**
- Original test data: **92.588%**
- Transformed test data: **83.86%**

### 1. Did Augmentation Improve Transformed Test Performance?

**Yes, significantly.** The augmented model achieved **89.116%** accuracy on the transformed test set, compared to **83.86%** for the baseline model. This represents an improvement of **+5.256 percentage points**.

This improvement demonstrates that training on augmented data (original + 5,000 transformed examples) helped the model learn to handle synonym replacements and typos more effectively. The model became more robust to the specific transformations used in the OOD evaluation.

### 2. Impact on Original Test Accuracy

**Slight improvement.** The augmented model achieved **92.764%** accuracy on the original test set, compared to **92.588%** for the baseline model. This represents a small improvement of **+0.176 percentage points**.

**Analysis:**
- The augmentation did not harm performance on clean, unmodified text.
- The model maintained its ability to handle original text while improving on transformed text.
- This suggests the augmented examples acted as a form of regularization, helping the model generalize better.
- The slight improvement indicates that the augmented data provided useful signal without overwhelming the original training data.

### 3. Intuitive Explanation of Results

**Why augmentation improved transformed performance:**
1. **Exposure to variations**: By training on transformed examples, the model learned that different surface forms (synonyms, typos) can represent the same sentiment.
2. **Invariant feature learning**: The model learned to extract features that are robust to these transformations, focusing on semantic content rather than exact word matches.
3. **More training data**: The additional 5,000 examples increased the training set from 25,000 to 30,000, providing more signal for learning.
4. **Distribution alignment**: Training on transformed data aligns the training distribution closer to the transformed test distribution, reducing the distribution gap.

**Why original performance slightly improved:**
1. **Regularization effect**: The augmented examples act as a form of data augmentation regularization, preventing overfitting to exact word patterns.
2. **Better generalization**: Learning to handle variations makes the model more robust overall, improving generalization to clean text as well.
3. **Semantic understanding**: The model learns to focus on semantic meaning rather than memorizing exact word sequences, which helps with both clean and transformed text.

**Why the improvement is significant for transformed but small for original:**
- The transformed test set has a large distribution gap from the original training data (8.728 point drop in baseline).
- Augmentation directly addresses this gap by training on similar transformations.
- The original test set already had high performance (92.588%), leaving less room for improvement.
- The model was already well-optimized for clean text, so the benefit is smaller.

### 4. One Limitation of This Augmentation Approach

**The augmentation only addresses the specific transformation used (synonym replacement + typos).** This approach may not generalize to other types of out-of-distribution (OOD) scenarios, such as:

1. **Different transformation types**: The model may not handle other OOD scenarios like:
   - Different writing styles or formality levels
   - Different domains (e.g., movie reviews vs. product reviews)
   - Different languages or dialects
   - Grammatical errors or word order changes
   - Adversarial perturbations

2. **Unknown distribution shifts**: In practice, OOD data can come from many unexpected sources. This augmentation method only helps with the specific transformations we anticipated and included in training.

3. **Overfitting to specific transformations**: The model might become too specialized to the exact transformation patterns used during augmentation, potentially reducing robustness to other types of variations.

4. **Limited coverage**: With only 5,000 augmented examples, we may not cover all possible transformation patterns, leaving some gaps in robustness.

**In summary**, this augmentation approach is effective for the specific transformations it targets, but it requires knowing the types of OOD scenarios in advance. It may not help with unexpected or different types of distribution shifts that could occur in real-world deployment.

---

## Summary Table

| Model | Original Test | Transformed Test | Accuracy Drop |
|-------|---------------|------------------|---------------|
| Baseline | 92.588% | 83.86% | 8.728 pts |
| Augmented | 92.764% | 89.116% | 3.648 pts |
| **Improvement** | **+0.176 pts** | **+5.256 pts** | **-5.08 pts** |

### Key Takeaways

1. **Augmentation significantly improved transformed test performance** (+5.256 points), demonstrating the effectiveness of training on transformed data.
2. **Original test performance slightly improved** (+0.176 points), showing that augmentation doesn't harm clean text performance.
3. **The accuracy gap decreased** from 8.728 points to 3.648 points, making the model more robust overall.
4. **The model became more robust** to the specific transformations used, while maintaining performance on original text.

