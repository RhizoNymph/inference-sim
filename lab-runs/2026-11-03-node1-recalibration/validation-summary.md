| dataset | role | old prefill / step / e2e | new prefill / step / e2e | new-static512 prefill / step / e2e |
|---|---|---|---|---|
| node1 static PP=1 | scalars in-sample | 53.6% / 4.3% / 12.9% | 9.7% / 1.7% / 3.0% | 1.7% / 1.7% / 2.7% |
| node1 prefill sweep | curve in-sample | 30.1% / 4.3% / 4.9% | 10.1% / 4.7% / 8.6% | 13.3% / 4.7% / 8.7% |
| node1 long context 4k-8k | held out | 51.6% / 4.0% / 6.0% | 0.8% / 0.9% / 8.5% | 0.8% / 0.9% / 8.5% |
| node1 long context 16k, 2x8k, 4x4k | held out | 50.0% / 3.6% / 18.1% | 0.8% / 7.5% / 6.4% | 0.8% / 7.5% / 6.4% |
| node1 decode sweep 1-32 x 512 | held out | 42.0% / 10.2% / 4.1% | 12.1% / 12.6% / 11.9% | 8.7% / 12.6% / 12.0% |
| node0 static PP=1 (transfer) | held out, other node | 68.8% / 6.0% / 20.9% | 19.7% / 1.6% / 4.9% | 10.8% / 1.6% / 4.6% |
