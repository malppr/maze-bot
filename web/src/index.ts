// maze-bot/web: browser port of the Wheely's maze sim and policies. mountMazeDemo arrives in M4.
export * from './sim/params';
export * from './sim/geometry';
export * from './sim/grid';
export * from './sim/sim';
export * from './sim/presets';
export { MLPPolicy, type WeightsJson, type PolicyOutput } from './policy/mlp';
export { HeuristicPolicy, type RuleTrace } from './policy/heuristic';
