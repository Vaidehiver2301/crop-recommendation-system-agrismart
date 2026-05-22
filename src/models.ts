// Client-Side Agronomic ML Classifiers (KNN, Decision Tree, Random Forest)

export interface Neighbor {
  index: number;
  label: string;
  distance: number;
  features: number[];
}

export interface DecisionNode {
  featureIdx?: number;     // 0: N, 1: P, 2: K, 3: Temp, 4: pH, 5: Rainfall
  threshold?: number;
  gini: number;
  samples: number;
  distribution: Record<string, number>;
  label?: string;          // Defined if leaf node
  left?: DecisionNode;
  right?: DecisionNode;
}

export interface ModelMetrics {
  accuracy: number;
  testPredictions: string[];
  testActuals: string[];
}

// Map column index to humand-readable label
export const FEATURE_NAMES = ["N", "P", "K", "Temperature", "pH", "Rainfall"];

// Seeded random helper for reproducible bootstrapping
function seededSample(seed: number, limit: number): number {
  const x = Math.sin(seed) * 10000;
  const rnd = x - Math.floor(x);
  return Math.floor(rnd * limit);
}

// ------------------------------------------------------------------
// 1. DATA PREPROCESSING (NORMALIZATION)
// ------------------------------------------------------------------
export interface ScalerParams {
  mins: number[];
  maxs: number[];
}

export function computeScaler(X: number[][]): ScalerParams {
  const numFeatures = X[0].length;
  const mins = Array(numFeatures).fill(Infinity);
  const maxs = Array(numFeatures).fill(-Infinity);

  for (let i = 0; i < X.length; i++) {
    for (let j = 0; j < numFeatures; j++) {
      if (X[i][j] < mins[j]) mins[j] = X[i][j];
      if (X[i][j] > maxs[j]) maxs[j] = X[i][j];
    }
  }

  return { mins, maxs };
}

export function scaleSample(sample: number[], scaler: ScalerParams): number[] {
  return sample.map((val, idx) => {
    const min = scaler.mins[idx];
    const max = scaler.maxs[idx];
    if (max === min) return 0;
    return (val - min) / (max - min);
  });
}

// ------------------------------------------------------------------
// 2. K-NEAREST NEIGHBORS (KNN) CLASSIFIER
// ------------------------------------------------------------------
export function predictKNN(
  X_train: number[][],
  y_train: string[],
  userSample: number[],
  k: number,
  scaler: ScalerParams
): { prediction: string; neighbors: Neighbor[] } {
  // Normalize both train and test point
  const scaledUser = scaleSample(userSample, scaler);
  const scaledTrain = X_train.map((row) => scaleSample(row, scaler));

  // Compute Euclidean Distances
  const list: Neighbor[] = [];
  for (let i = 0; i < scaledTrain.length; i++) {
    let sumSq = 0;
    for (let j = 0; j < scaledUser.length; j++) {
      sumSq += Math.pow(scaledUser[j] - scaledTrain[i][j], 2);
    }
    const dist = Math.sqrt(sumSq);
    list.push({
      index: i,
      label: y_train[i],
      distance: dist,
      features: X_train[i]
    });
  }

  // Sort by smallest distance
  list.sort((a, b) => a.distance - b.distance);

  // Take top K neighbors
  const neighbors = list.slice(0, k);

  // Count occurrences
  const counts: Record<string, number> = {};
  neighbors.forEach((n) => {
    counts[n.label] = (counts[n.label] || 0) + 1;
  });

  // Find majority vote
  let bestLabel = "";
  let maxCount = -1;
  let minDistForTie = Infinity;

  Object.keys(counts).forEach((label) => {
    const count = counts[label];
    if (count > maxCount) {
      maxCount = count;
      bestLabel = label;
      // Get distance of first neighbor with this label as ties breaker
      const firstOccur = neighbors.find((n) => n.label === label);
      minDistForTie = firstOccur ? firstOccur.distance : Infinity;
    } else if (count === maxCount) {
      // Tie breaker: pick label with the single closest neighbor
      const firstOccur = neighbors.find((n) => n.label === label);
      if (firstOccur && firstOccur.distance < minDistForTie) {
        bestLabel = label;
        minDistForTie = firstOccur.distance;
      }
    }
  });

  return { prediction: bestLabel, neighbors };
}

// ------------------------------------------------------------------
// 3. DECISION TREE CLASSIFIER
// ------------------------------------------------------------------
function calculateGini(labels: string[]): number {
  const n = labels.length;
  if (n === 0) return 0;
  const counts: Record<string, number> = {};
  labels.forEach((l) => { counts[l] = (counts[l] || 0) + 1; });
  let sumSq = 0;
  Object.keys(counts).forEach((key) => {
    const p = counts[key] / n;
    sumSq += p * p;
  });
  return 1 - sumSq;
}

export function buildDecisionTree(
  X: number[][],
  y: string[],
  depth: number,
  maxDepth: number,
  featureSubsetCount?: number // Random feature subset size for Random Forest
): DecisionNode {
  const samples = X.length;

  // Build label distribution
  const distribution: Record<string, number> = {};
  y.forEach((label) => {
    distribution[label] = (distribution[label] || 0) + 1;
  });

  const nodeGini = calculateGini(y);

  // Find majority class label for leaf prediction
  let majorityLabel = "";
  let maxCount = -1;
  Object.keys(distribution).forEach((key) => {
    if (distribution[key] > maxCount) {
      maxCount = distribution[key];
      majorityLabel = key;
    }
  });

  // Check leaf stopping conditions
  if (depth >= maxDepth || samples < 2 || nodeGini === 0) {
    return {
      gini: nodeGini,
      samples,
      distribution,
      label: majorityLabel
    };
  }

  // Choose feature indices to evaluate
  const numFeatures = X[0].length;
  let featuresToTry = Array.from({ length: numFeatures }, (_, i) => i);

  if (featureSubsetCount && featureSubsetCount < numFeatures) {
    // Basic deterministic shuffle based on a seed derived from depth & samples
    let s = depth * 31 + samples;
    const shuffled = [...featuresToTry];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = seededSample(s++, i + 1);
      const tmp = shuffled[i];
      shuffled[i] = shuffled[j];
      shuffled[j] = tmp;
    }
    featuresToTry = shuffled.slice(0, featureSubsetCount);
  }

  let bestGain = -1;
  let bestFeatureIdx = -1;
  let bestThreshold = -1;
  let bestLeftIdxs: number[] = [];
  let bestRightIdxs: number[] = [];

  // Find optimal split points
  featuresToTry.forEach((fIdx) => {
    // Sort values for evaluation thresholds
    const values = X.map((row) => row[fIdx]);
    const sortedVals = Array.from(new Set(values)).sort((a, b) => a - b);

    // Try splits between adjacent values
    for (let idx = 0; idx < sortedVals.length - 1; idx++) {
      const threshold = (sortedVals[idx] + sortedVals[idx + 1]) / 2;

      const leftIdxs: number[] = [];
      const rightIdxs: number[] = [];

      for (let sIdx = 0; sIdx < samples; sIdx++) {
        if (X[sIdx][fIdx] < threshold) {
          leftIdxs.push(sIdx);
        } else {
          rightIdxs.push(sIdx);
        }
      }

      if (leftIdxs.length === 0 || rightIdxs.length === 0) continue;

      const leftLabels = leftIdxs.map((i) => y[i]);
      const rightLabels = rightIdxs.map((i) => y[i]);

      const leftGini = calculateGini(leftLabels);
      const rightGini = calculateGini(rightLabels);

      const weightedGini = (leftIdxs.length / samples) * leftGini + (rightIdxs.length / samples) * rightGini;
      const gain = nodeGini - weightedGini;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeatureIdx = fIdx;
        bestThreshold = threshold;
        bestLeftIdxs = leftIdxs;
        bestRightIdxs = rightIdxs;
      }
    }
  });

  // If no improvement can be made, convert to leaf node
  if (bestGain <= 0) {
    return {
      gini: nodeGini,
      samples,
      distribution,
      label: majorityLabel
    };
  }

  // Recursively train child branches
  const leftX = bestLeftIdxs.map((i) => X[i]);
  const leftY = bestLeftIdxs.map((i) => y[i]);
  const rightX = bestRightIdxs.map((i) => X[i]);
  const rightY = bestRightIdxs.map((i) => y[i]);

  return {
    gini: nodeGini,
    samples,
    distribution,
    featureIdx: bestFeatureIdx,
    threshold: bestThreshold,
    left: buildDecisionTree(leftX, leftY, depth + 1, maxDepth, featureSubsetCount),
    right: buildDecisionTree(rightX, rightY, depth + 1, maxDepth, featureSubsetCount)
  };
}

export function predictTree(node: DecisionNode, sample: number[]): string {
  if (node.label !== undefined) {
    return node.label;
  }
  if (node.featureIdx === undefined || node.threshold === undefined) {
    return "Rice"; // Fallback node prediction
  }
  const featureVal = sample[node.featureIdx];
  if (featureVal < node.threshold) {
    return predictTree(node.left!, sample);
  } else {
    return predictTree(node.right!, sample);
  }
}

// ------------------------------------------------------------------
// 4. RANDOM FOREST CLASSIFIER
// ------------------------------------------------------------------
export interface RandomForest {
  trees: DecisionNode[];
}

export function trainRandomForest(
  X: number[][],
  y: string[],
  numTrees: number,
  maxDepth: number
): RandomForest {
  const trees: DecisionNode[] = [];
  const sampleCount = X.length;
  let seed = 12345;

  for (let t = 0; t < numTrees; t++) {
    // Generate Bootstrap Sample (Sample with replacement)
    const bootX: number[][] = [];
    const bootY: string[] = [];

    for (let s = 0; s < sampleCount; s++) {
      const idx = seededSample(seed++, sampleCount);
      bootX.push(X[idx]);
      bootY.push(y[idx]);
    }

    // Train a Randomized Tree choosing subset of features e.g. sqrt(6) ~ 3 features
    const tree = buildDecisionTree(bootX, bootY, 0, maxDepth, 3);
    trees.push(tree);
  }

  return { trees };
}

export interface ForestVote {
  crop: string;
  votes: number;
  percentage: number;
}

export function predictForest(
  forest: RandomForest,
  sample: number[]
): { prediction: string; treePredictions: string[]; voteBreakdown: ForestVote[] } {
  const treePredictions = forest.trees.map((tree) => predictTree(tree, sample));

  const counts: Record<string, number> = {};
  treePredictions.forEach((p) => {
    counts[p] = (counts[p] || 0) + 1;
  });

  const voteBreakdown: ForestVote[] = Object.keys(counts).map((crop) => ({
    crop,
    votes: counts[crop],
    percentage: Math.round((counts[crop] / forest.trees.length) * 100)
  })).sort((a, b) => b.votes - a.votes);

  const prediction = voteBreakdown.length > 0 ? voteBreakdown[0].crop : "Rice";

  return {
    prediction,
    treePredictions,
    voteBreakdown
  };
}

// ------------------------------------------------------------------
// 5. EVALUATE CLASSIFIER UTILS
// ------------------------------------------------------------------
export function evaluateKNN(
  X_train: number[][],
  y_train: string[],
  X_test: number[][],
  y_test: string[],
  k: number,
  scaler: ScalerParams
): ModelMetrics {
  const testPredictions = X_test.map((row) => predictKNN(X_train, y_train, row, k, scaler).prediction);
  let hits = 0;
  for (let i = 0; i < X_test.length; i++) {
    if (testPredictions[i] === y_test[i]) hits++;
  }
  return {
    accuracy: hits / X_test.length,
    testPredictions,
    testActuals: y_test
  };
}

export function evaluateDecisionTree(
  tree: DecisionNode,
  X_test: number[][],
  y_test: string[]
): ModelMetrics {
  const testPredictions = X_test.map((row) => predictTree(tree, row));
  let hits = 0;
  for (let i = 0; i < X_test.length; i++) {
    if (testPredictions[i] === y_test[i]) hits++;
  }
  return {
    accuracy: hits / X_test.length,
    testPredictions,
    testActuals: y_test
  };
}

export function evaluateRandomForest(
  forest: RandomForest,
  X_test: number[][],
  y_test: string[]
): ModelMetrics {
  const testPredictions = X_test.map((row) => predictForest(forest, row).prediction);
  let hits = 0;
  for (let i = 0; i < X_test.length; i++) {
    if (testPredictions[i] === y_test[i]) hits++;
  }
  return {
    accuracy: hits / X_test.length,
    testPredictions,
    testActuals: y_test
  };
}
