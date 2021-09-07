# Multi-Agent Cooperation in Hanabi with Policy Optimisation

This is the source code for my master's dissertation project _Multi-Agent Cooperation in Hanabi with Policy Optimisation_.

The implementation is based on [Proximal Policy Optimisation](https://arxiv.org/abs/1707.06347) and [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment).

## Requirement

- Python 3.7+ (because I can't survive without f-strings)
- PyTorch (currently CPU only) (someone buy me a NVIDIA laptop plz?)
- [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment)

## Performance

### Hanabi-Small

Hanabi-Small is a smaller version of Hanabi with a maximum score of 10. Currently we can train an agent on Hanabi-Small in under 8 hours on an 8-core CPU machine.

<object data="figures/mlp_training.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="figures/mlp_training.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="figures/mlp_training.pdf">Download PDF</a>.</p>
    </embed>
</object>

<object data="figures/rnn_training.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="figures/rnn_training.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="figures/rnn_training.pdf">Download PDF</a>.</p>
    </embed>
</object>
