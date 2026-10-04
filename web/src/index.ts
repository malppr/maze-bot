// maze-bot/web: the Wheely's maze demo (mountMazeDemo) plus the sim and policies it runs on.
export * from './sim/params';
export * from './sim/geometry';
export * from './sim/grid';
export * from './sim/sim';
export * from './sim/presets';
export { MLPPolicy, type WeightsJson, type PolicyOutput } from './policy/mlp';
export { HeuristicPolicy, type RuleTrace } from './policy/heuristic';
export { mountMazeDemo, type MazeDemoOptions, type MazeDemoHandle } from './demo/mount';
