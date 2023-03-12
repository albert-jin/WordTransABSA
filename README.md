# WordTransABSA
Target Word Transferred Language Model for tackling ABSA (WordTransABSA)

:star: :star: :star: :star: :star:
> Word Transferred Language Model for Aspect-based Sentiment Analysis]{WordTransABSA: A Novel BERT-based Aspect-based Sentiment Analysis Approach by Utilizing ``Masked Language Model'' to Predict Affective Tokens for Target Words :smiley:

[![](/assets/wordtransabsa.pdf "Model Architecture of WordTransABSA")][Model Architecture of WordTransABSA]

```angular2html
> In order to alleviate the limitations of mainstream fine-tuning methods, we propose Target Word Transferred ABSA (WordTransABSA). WordTransABSA subverts the conventional criterion of Transformerbased fine-tuning methods by utilizing the entire parameters in Transformer, to fully exploit the prior knowledge in Transformer.

> Compared with prompt learning, which requiresconstructing extra pre-defined templates with specific slots to better invoke the Pre-train LMs, WordTransABSA only needs sentiment-related pivot tokens to obtain the sentiment polarity-related affective tokens on the position of an aspect term.

> In the specific WordTransABSA implementation, we explore different discover measures to search high quality sentiment-related pivot tokens and try additional transferred word optimization strategies to stimulate the semantic understanding potential of PLMs better.

> The extensive experiments under the data-sufficient scenario (full-data supervised learning) and data-scarce scenario (few-shot learning) validate the superiority and effectiveness of the WordTransABSA, suggesting that regressing to the Transformer pretraining paradigm is a better solution for some specific scenarios like ABSA.
```

***

> The ABSA experimental performance statistics (% Accuracy) between the WordsTransABSAs and the SOTA baselines.

[![](/assets/comparisons.PNG "comparisons")][comparisons]

***

> The performance statistics (% Accuracy) of our WordTransABSA variants and the SOTA baselines under the different
few-shot learning settings. All results are averaged over 10 runs to maintain the experimental authenticity.

[![](/assets/fewshotcomparisons.PNG "fewshot_comparisons")][fewshot_comparisons]

***

# Thanks!

### Star and Fork. 

:sunglasses: :pray: :innocent: :heartpulse: :heartpulse: :heartpulse: 	:stuck_out_tongue_closed_eyes:
