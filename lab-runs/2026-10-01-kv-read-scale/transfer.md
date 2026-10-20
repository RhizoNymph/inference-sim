| regime | variant | prefill | decode step | end-to-end |
|---|---|---:|---:|---:|
| 7B PP=1, 5 shapes (node0) | before | 0.8% | 0.5% | 0.9% |
| 7B PP=1, 5 shapes (node0) | kv | 0.8% | 2.7% | 2.1% |
| 7B PP=2 | before | 10.4% | 3.8% | 1.3% |
| 7B PP=2 | kv | 10.4% | 2.8% | 2.1% |
| 14B PP=2 (model transfer) | before | 8.7% | 7.2% | 5.1% |
| 14B PP=2 (model transfer) | kv | 8.7% | 6.2% | 5.3% |

| serving variant | TTFT p50 | TPOT p50 | ITL p50 | E2EL p50 | tok/s | TPOT p50 at 4/6/8 req/s |
|---|---:|---:|---:|---:|---:|---|
| before | 17.7% | 6.5% | 2.9% | 7.1% | 8.4% | -6.1% / -15.8% / -12.7% |
| kv | 15.5% | 4.7% | 3.7% | 3.8% | 7.3% | -3.4% / -7.7% / -10.5% |
