# Scientific report

## 1. Paper summary

## 2. Method

## 3. Experimental setup

We use a deterministic synthetic sequence-reversal task. Input tokens are sampled
uniformly from a vocabulary of 29 data symbols; token 0 is reserved for padding,
token 1 begins decoder input (`BOS`), and token 2 is an explicit end-of-sequence
marker (`EOS`). The marker makes the
length observable when examples of different lengths share a padded batch. The
training set contains 10,000 sequences, while the validation and in-distribution
test sets contain 1,000 sequences each. Their lengths are sampled uniformly from
5 to 20 tokens. A separate 1,000-example test set contains lengths 21–40 and is
used only to measure out-of-distribution length generalization. The four splits
are generated from independent, recorded pseudorandom seeds.

The primary model is a two-layer encoder–decoder Transformer with sinusoidal
positional encoding, model width 128, feed-forward width 256, ReLU activation,
and dropout 0.1. A two-layer encoder–decoder LSTM with the same embedding and
hidden width is used as a baseline. Both models use teacher forcing during
training and greedy autoregressive decoding during evaluation. For the ablation,
the Transformer uses 1, 2, 4, or 8 heads
while keeping `d_model=128`; therefore the projection parameter count is fixed
and only the per-head width changes.

Models are trained with AdamW, learning rate 0.001, weight decay 0.0001, batch
size 64, cross-entropy loss ignoring padding, and gradient norm clipping at 1.0.
Training lasts at most 20 epochs and stops after five epochs without improvement
in teacher-forced validation loss. Autoregressive decoding is deliberately
reserved for the final validation and test passes to keep the sweep feasible.
Final configurations are run with seeds 13, 37, and 71. We report token accuracy, exact-sequence accuracy,
trainable parameter count, and wall-clock training time. Raw per-epoch histories,
checkpoints, configuration, software versions, and hardware metadata are saved
under `artifacts/`.

The final sweep was executed on Windows 11 using Python 3.12.13 and CPU-only
PyTorch 2.12.1 on 16 logical CPU threads. Individual LSTM runs took about
5.7 minutes on average, while Transformer runs took approximately 10.5–12.1
minutes on average depending on head count; the longest run remained well below
the four-hour hardware constraint.

These settings are a deliberate laptop-scale deviation from Vaswani et al.
(2017): the original WMT translation datasets and base/large models are replaced
by a controlled algorithmic task and a tiny model. The head-count ablation keeps
the central comparison of differently partitioned attention representations,
but its absolute results should not be interpreted as translation quality.

During protocol development we piloted a simpler encoder-only token classifier.
It plateaued at approximately 23% token accuracy and 0% exact-sequence accuracy,
even after adding an explicit `EOS` token. We rejected that simplification before
the multi-seed sweep and adopted the encoder–decoder protocol above.

## 4. Results

## 5. Ablation: number of attention heads

## 6. Limitations and reproducibility notes

## 7. Conclusions

## References
