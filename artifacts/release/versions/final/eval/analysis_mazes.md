# mazes: mix_fwd5mem_12x12_pcrit_s0 vs baseline (1000 test maps)

Overall: ours 59.7% vs baseline 68.6%
Both solve 505 | only baseline 181 | only ours 92 | neither 222

## path length (units)

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-5 | 174 | 91% | 87% |
| 5-8 | 265 | 80% | 81% |
| 8-12 | 263 | 60% | 72% |
| 12-16 | 157 | 36% | 54% |
| 16+ | 141 | 7% | 33% |

## detour factor (path / straight line)

| bin | maps | ours | baseline |
|---|---|---|---|
| 1-1.3 | 333 | 89% | 85% |
| 1.3-1.8 | 264 | 62% | 73% |
| 1.8-2.5 | 167 | 44% | 62% |
| 2.5-4 | 147 | 32% | 47% |
| 4+ | 89 | 18% | 43% |

## turns along the path

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-1 | 407 | 84% | 84% |
| 2-3 | 346 | 58% | 66% |
| 4-5 | 166 | 31% | 52% |
| 6-8 | 64 | 5% | 44% |
| 9+ | 17 | 0% | 18% |

## corridor clearance (narrowest 10%)

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-0.7 | 504 | 48% | 62% |
| 0.7-0.8 | 376 | 67% | 72% |
| 0.8-0.9 | 80 | 84% | 86% |
| 0.9-1 | 12 | 92% | 75% |
| 1+ | 28 | 96% | 93% |

## line of sight

| | maps | ours | baseline |
|---|---|---|---|
| B visible from A | 95 | 94% | 91% |
| B hidden from A | 905 | 56% | 66% |

## our failures

- looping: 249 (24.9% of maps), median progress before failing -4% of the path
- stuck: 70 (7.0% of maps), median progress before failing 3% of the path
- progressing: 84 (8.4% of maps), median progress before failing 14% of the path
- where it gives out: <25% of the way 84%, 25-75% 15%, >75% 1% (of failures)
