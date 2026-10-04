# mazes: mix_fwd5mem_12x12x4_g995_s0 vs baseline (1000 test maps)

Overall: ours 53.3% vs baseline 68.6%
Both solve 456 | only baseline 230 | only ours 77 | neither 237

## path length (units)

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-5 | 174 | 87% | 87% |
| 5-8 | 265 | 78% | 81% |
| 8-12 | 263 | 47% | 72% |
| 12-16 | 157 | 24% | 54% |
| 16+ | 141 | 9% | 33% |

## detour factor (path / straight line)

| bin | maps | ours | baseline |
|---|---|---|---|
| 1-1.3 | 333 | 85% | 85% |
| 1.3-1.8 | 264 | 59% | 73% |
| 1.8-2.5 | 167 | 32% | 62% |
| 2.5-4 | 147 | 20% | 47% |
| 4+ | 89 | 11% | 43% |

## turns along the path

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-1 | 407 | 80% | 84% |
| 2-3 | 346 | 50% | 66% |
| 4-5 | 166 | 20% | 52% |
| 6-8 | 64 | 3% | 44% |
| 9+ | 17 | 0% | 18% |

## corridor clearance (narrowest 10%)

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-0.7 | 504 | 38% | 62% |
| 0.7-0.8 | 376 | 64% | 72% |
| 0.8-0.9 | 80 | 76% | 86% |
| 0.9-1 | 12 | 83% | 75% |
| 1+ | 28 | 96% | 93% |

## line of sight

| | maps | ours | baseline |
|---|---|---|---|
| B visible from A | 95 | 91% | 91% |
| B hidden from A | 905 | 49% | 66% |

## our failures

- looping: 378 (37.8% of maps), median progress before failing 2% of the path
- stuck: 31 (3.1% of maps), median progress before failing 1% of the path
- progressing: 58 (5.8% of maps), median progress before failing 21% of the path
- where it gives out: <25% of the way 81%, 25-75% 16%, >75% 3% (of failures)
