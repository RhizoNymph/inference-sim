| regime | variant | prefill | decode step | end-to-end |
|---|---|---:|---:|---:|
| prefill token sweep (1x16-1x4096) | before | 30.1% | 0.7% | 1.1% |
| prefill token sweep (1x16-1x4096) | floor | 10.9% | 0.7% | 1.2% |
| prefill token sweep (1x16-1x4096) | curve | 3.2% | 0.7% | 1.4% |
| 7B PP=1 (5 shapes) | before | 4.3% | 0.5% | 1.5% |
| 7B PP=1 (5 shapes) | floor | 4.3% | 0.5% | 1.5% |
| 7B PP=1 (5 shapes) | curve | 0.8% | 0.5% | 0.9% |
| 7B long context 4k-16k (node1) | before | 1.4% | 2.5% | 1.8% |
| 7B long context 4k-16k (node1) | floor | 1.4% | 2.5% | 1.8% |
| 7B long context 4k-16k (node1) | curve | 4.2% | 2.5% | 3.1% |
| 7B decode batch 1-32 (node1) | before | 5.7% | 2.7% | 2.4% |
| 7B decode batch 1-32 (node1) | floor | 5.7% | 2.7% | 2.4% |
| 7B decode batch 1-32 (node1) | curve | 6.0% | 2.7% | 3.0% |
| 7B PP=2 | before | 14.6% | 3.8% | 1.6% |
| 7B PP=2 | floor | 14.6% | 3.8% | 1.6% |
| 7B PP=2 | curve | 10.4% | 3.8% | 1.3% |
| 14B PP=2 | before | 13.4% | 7.2% | 4.9% |
| 14B PP=2 | floor | 13.4% | 7.2% | 4.9% |
| 14B PP=2 | curve | 8.7% | 7.2% | 5.1% |

| serving variant | TTFT p50 | TPOT p50 | ITL p50 | E2EL p50 | tok/s | TTFT p50 at 1/2/4 req/s |
|---|---:|---:|---:|---:|---:|---|
| before | 29.8% | 10.8% | 3.3% | 12.7% | 9.0% | -33% / -35% / -39% |
| floor | 29.8% | 10.8% | 3.3% | 12.7% | 9.0% | -33% / -35% / -39% |
| curve | 21.3% | 6.5% | 2.9% | 7.3% | 8.5% | -23% / -25% / -26% |
| curve+frontend | 17.7% | 6.5% | 2.9% | 7.1% | 8.4% | -15% / -17% / -19% |
