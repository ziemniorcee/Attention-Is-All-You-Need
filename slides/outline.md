# Presentation outline

1. Research question
2. Paper and headline claim
3. Transformer architecture
4. Controlled reverse task
5. Experimental protocol
   - Deterministic reverse task; train/validation/test lengths 5–20.
   - Separate length-generalization test at lengths 21–40.
   - Teacher forcing in training; greedy autoregressive decoding in evaluation.
   - Three seeds; token and exact-sequence accuracy; fixed training budget.
   - AdamW, up to 20 epochs, early stopping, best validation checkpoint.
6. Transformer vs LSTM
7. Attention-head ablation
8. Generalization to longer sequences
9. Reproducibility and limitations
10. Conclusions
