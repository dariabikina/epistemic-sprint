## One question from start to finish (uncompressed Qwen3-8B)

Every question has one correct answer that we know. Three things are done with it.

**1. The model answers the question.** After three solved examples that show the format, it gets:

> Answer with only this person's name, without explanation. Who is the author of The Bicentennial Man?

It writes **Isaac Asimov**, which is correct. This question counts as *answered correctly*.

**2. We read how probable the model finds each answer on a list.** For this question the list has 22 names: the correct answer in its spellings, and wrong answers chosen to be plausible (other authors). The model does not write anything here. For each name we compute the probability that the model would give it as its answer to the prompt above.

| answer on the list | probability |
|---|---|
| Isaac Asimov (correct) | 98% |
| Lewis Carroll (the most probable wrong answer) | 1 in 57 million |

The correct answer is more probable than every wrong one, so the model *knows* this fact. The *answer margin* says by how much: ln(0.98) − ln(probability of Lewis Carroll) = **17.8**. A margin below zero means a wrong answer is more probable than the correct one.

**3. Separately, we ask the model whether it knows.** In a new conversation with no examples it gets:

> Do you know the answer to this question? Who is the author of The Bicentennial Man? Answer with just one word, Yes or No.

Again it does not write. We read the probability of its next word: Yes 87%, No 11%, anything else 2%. Between Yes and No that is **P(Yes) = 0.89**.

**The same three steps for two more questions**

| question | the model wrote | correct answer | answer margin | P(Yes) |
|---|---|---|---|---|
| Who is the author of The Bicentennial Man? | Isaac Asimov (correct) | Isaac Asimov | +17.8 | 0.89 |
| Who is the author of The Hostile Hospital? | John Wyndham (wrong) | Lemony Snicket | −2.6 | 0.33 |
| Who is the author of The Widow of Kessering Street? (an invented book) | John Grisham | none exists | not defined | 0.19 |

**How the figures use these numbers**

- *Does P(Yes) go with a correct answer?* Pair a correctly answered question with a wrongly answered one, such as the first two rows. P(Yes) is 0.89 against 0.33: higher for the correctly answered one, so this pair is ordered correctly. Over all such pairs in English, P(Yes) orders 73 of 100 correctly and the answer margin 85. Chance gives 50.
- *Does the model say No when there is nothing to know?* For invented subjects the average P(Yes) is 0.20; for real ones 0.57.
- *What happens after compression?* The same model pruned with WANDA 2:4 answers the first question with Johnathan Swift. It now gives Isaac Asimov a probability of 1 in 2 million, less than Lewis Carroll (1 in 6 thousand): the fact is *lost*. Its P(Yes) for this question fell from 0.89 to 0.11, which looks as if it noticed. But P(Yes) fell on the facts it kept as well: from 0.53 to 0.21 on average for lost facts, and from 0.69 to 0.37 for kept ones. The third figure tests whether the reply separates the two groups.
