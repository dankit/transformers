This is my personal implementation of the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762). All credit goes to the original authors.

This project was meant to give me an intuitive understanding of transformers and PyTorch by applying the theory in practice.

AI assistants were used only to double-check the implementation, not to write the transformer itself. They helped catch minor syntax issues, bugs, and served as a form of peer review.

To my knowledge, modern transformers differ in several ways:

-Sinusoidal encodings are often replaced with rotary positional embeddings (RoPE).

-SwiGLU has largely replaced ReLU.

-Attention applies dropout after the QK<sup>T</sup> matrix, not on the final output. (I kept the original paper’s design here.)

-Multi-head attention has several variants, including multi-query attention, grouped-query attention, and multi-head latent attention.

-Pre-LayerNorm or RMSNorm is now common.

-Mixture-of-experts architectures are increasingly used.

These are all ideas I hope to explore in a future iteration of my transformer. Feel free to reach out with any comments or bug fixes.


update 1/1/2026:
I've added swiglu, rmsnorm, rotary positional embeddings.