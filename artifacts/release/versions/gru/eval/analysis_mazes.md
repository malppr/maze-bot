# mazes: mix_fwd5mem_gru8x4_s0 vs baseline (1000 test maps)

Overall: ours 54.7% vs baseline 68.6%
Both solve 475 | only baseline 211 | only ours 72 | neither 242

## path length (units)

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-5 | 174 | 93% | 87% |
| 5-8 | 265 | 74% | 81% |
| 8-12 | 263 | 53% | 72% |
| 12-16 | 157 | 26% | 54% |
| 16+ | 141 | 6% | 33% |

## detour factor (path / straight line)

| bin | maps | ours | baseline |
|---|---|---|---|
| 1-1.3 | 333 | 87% | 85% |
| 1.3-1.8 | 264 | 60% | 73% |
| 1.8-2.5 | 167 | 35% | 62% |
| 2.5-4 | 147 | 22% | 47% |
| 4+ | 89 | 8% | 43% |

## turns along the path

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-1 | 407 | 83% | 84% |
| 2-3 | 346 | 50% | 66% |
| 4-5 | 166 | 22% | 52% |
| 6-8 | 64 | 3% | 44% |
| 9+ | 17 | 0% | 18% |

## corridor clearance (narrowest 10%)

| bin | maps | ours | baseline |
|---|---|---|---|
| 0-0.7 | 504 | 41% | 62% |
| 0.7-0.8 | 376 | 64% | 72% |
| 0.8-0.9 | 80 | 76% | 86% |
| 0.9-1 | 12 | 75% | 75% |
| 1+ | 28 | 100% | 93% |

## line of sight

| | maps | ours | baseline |
|---|---|---|---|
| B visible from A | 95 | 94% | 91% |
| B hidden from A | 905 | 51% | 66% |

## our failures

- looping: 373 (37.3% of maps), median progress before failing 3% of the path
- progressing: 80 (8.0% of maps), median progress before failing 8% of the path
- where it gives out: <25% of the way 78%, 25-75% 19%, >75% 3% (of failures)
