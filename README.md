# Deep learning — projects and studies

Work from the IIT Kharagpur Executive PG Certificate in Applied AI and Machine
Learning, and the projects built around it. Each directory is self-contained:
its own README, its own results, its own reproduction commands.

## Projects

### [`har/`](har/) — Which sequence model do you ship to a wearable?

Four architectures (RNN, LSTM, GRU, 1-D CNN) trained on UCI HAR under matched
hyperparameters, three seeds each, measured in FP32 and INT8 for accuracy, size
and single-window latency. Framed as a deployment cost decision rather than a
leaderboard.

| | |
|---|---|
| **Headline** | The 1-D CNN runs **22× faster** than the GRU for an accuracy gap that sits inside the seed noise |
| **Counter-result** | INT8 quantisation cost **25× latency** to save 3.5× size on x86 CPU |
| **Gotcha found** | `quantize_dynamic` silently ignores `nn.RNN` and `Conv1d` — it accepts the request and changes nothing |
| **Discipline** | Four of six architecture pairs are reported as *indistinguishable*, because their gaps fall inside the seed spread |

[**Read the study →**](har/README.md)

## Licence

MIT — see [`LICENSE`](LICENSE). Applies to all projects in this repository.
