# Data processing for [NeuSum](https://github.com/magic282/NeuSum)

This repo contains the code which can generate the training data (CNN / Daily Mail) needed by [NeuSum](https://github.com/magic282/NeuSum).



1. Preprocess CNN/DM dataset using abisee's scripts: https://github.com/abisee/cnn-dailymail
2. Convert its output to the format shown in the `sample_data` folder. The format of files:
    1. File train.txt.src is the input document. Each line contains several tokenized sentences delimited by ##SENT## of a document.
    2. File train.txt.tgt is the summary of document. Each line contains several tokenized summaries delimited by ##SENT## of the corresponding document.

3. Use `find_oracle.py` to search the best sentences to be extracted. The arguments of the `main` functions are: `document_file`, `summary_file` and `output_path`.
4. Next, build the ROUGE score gain file using `get_mmr_regression_gain.py`. The usage is shown in the code entry.


## Note
The algorithm is a brute-force search, which can be slow in some cases. Therefore, running it in parallel is recommended (and it is what I did in my experiments).

