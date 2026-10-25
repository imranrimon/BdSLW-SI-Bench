# P2 full metric suite (held-out TEST)

Primary = single-model **3-seed mean ± std** (matches Table 2 convention). Top-1(ens) = 3-seed softmax ensemble.

| Arch | Top-1 (mean±std) | Top-1(ens) | Top-5 | Macro-F1 | MRR | R@10 | ECE% | NLL | Brier |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Adaptive GNN | 78.94±1.06 | 80.00 | 93.37 | 78.09 | 0.855 | 97.55 | 12.16 | 1.253 | 0.329 |
| ST-GCN | 74.51±0.39 | 76.63 | 91.43 | 72.64 | 0.823 | 95.34 | 12.17 | 1.373 | 0.389 |
| Attention GNN | 73.97±1.01 | 77.73 | 92.33 | 71.86 | 0.823 | 96.21 | 13.71 | 1.274 | 0.402 |
| SLGTFormer | 73.26±4.77 | 79.12 | 93.48 | 71.92 | 0.822 | 97.22 | 12.87 | 1.308 | 0.407 |
| GNN+Bi-LSTM | 71.58±3.35 | 76.04 | 91.92 | 70.04 | 0.806 | 96.51 | 14.72 | 1.469 | 0.443 |
| BlockGCN | 65.23±2.24 | 70.62 | 90.06 | 64.18 | 0.761 | 95.12 | 16.57 | 1.606 | 0.524 |
| Pose-LSTM | 61.64±1.21 | 64.10 | 88.35 | 60.34 | 0.734 | 94.46 | 18.83 | 1.910 | 0.579 |
| GNN+Transformer | 54.18±7.58 | 60.88 | 85.93 | 52.47 | 0.682 | 93.48 | 31.47 | 3.129 | 0.741 |

## Per-signer TEST accuracy (Top-1, ensemble)
- **Adaptive GNN**: U13 73.8 (n=621), U2 85.2 (n=744)
- **ST-GCN**: U13 69.4 (n=621), U2 82.7 (n=744)
- **Attention GNN**: U13 70.4 (n=621), U2 83.9 (n=744)
- **SLGTFormer**: U13 74.1 (n=621), U2 83.3 (n=744)
- **GNN+Bi-LSTM**: U13 63.3 (n=621), U2 86.7 (n=744)
- **BlockGCN**: U13 67.3 (n=621), U2 73.4 (n=744)
- **Pose-LSTM**: U13 53.5 (n=621), U2 73.0 (n=744)
- **GNN+Transformer**: U13 48.8 (n=621), U2 71.0 (n=744)

## Hardest classes (lowest Top-1, BlockGCN)
cls42:0%, cls22:9%, cls28:13%, cls13:17%, cls24:29%, cls48:35%, cls51:36%, cls0:37%
