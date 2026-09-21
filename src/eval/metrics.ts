import type { EvalItem, ArmResult, Metrics, Label } from '../types.js';

/**
 * Compute evaluation metrics for arm results.
 * Includes accuracy, ECE, Brier score, and per-label metrics.
 */

export function computeMetrics(items: EvalItem[], results: ArmResult[]): Metrics {
  const itemMap = new Map(items.map(item => [item.id, item]));
  
  let correct = 0;
  let total = 0;
  const perLabel = {
    answer: { correct: 0, total: 0 },
    refuse: { correct: 0, total: 0 },
    escalate: { correct: 0, total: 0 }
  };
  
  // For ECE calculation
  const bins = Array(10).fill(0).map(() => ({ correct: 0, total: 0, sumConf: 0 }));
  
  // For Brier score
  let brierSum = 0;
  
  for (const result of results) {
    if (result.error) continue;
    
    const item = itemMap.get(result.itemId);
    if (!item) continue;
    
    total++;
    const isCorrect = result.prediction.label === item.groundTruth;
    
    if (isCorrect) {
      correct++;
      perLabel[item.groundTruth].correct++;
    }
    perLabel[item.groundTruth].total++;
    
    // ECE: bin by confidence
    const confidence = result.prediction.confidence;
    const binIdx = Math.min(9, Math.floor(confidence * 10));
    bins[binIdx].total++;
    bins[binIdx].sumConf += confidence;
    if (isCorrect) {
      bins[binIdx].correct++;
    }
    
    // Brier score: sum of squared errors for all labels
    if (result.prediction.probabilities) {
      const probs = result.prediction.probabilities;
      const labels: Label[] = ['answer', 'refuse', 'escalate'];
      
      for (const label of labels) {
        const target = label === item.groundTruth ? 1 : 0;
        const predicted = probs[label];
        brierSum += Math.pow(predicted - target, 2);
      }
    } else {
      // Fallback if no probabilities: use confidence
      const labels: Label[] = ['answer', 'refuse', 'escalate'];
      for (const label of labels) {
        const target = label === item.groundTruth ? 1 : 0;
        const predicted = label === result.prediction.label ? confidence : (1 - confidence) / 2;
        brierSum += Math.pow(predicted - target, 2);
      }
    }
  }
  
  // Compute ECE
  let eceSum = 0;
  for (const bin of bins) {
    if (bin.total === 0) continue;
    const avgConf = bin.sumConf / bin.total;
    const accuracy = bin.correct / bin.total;
    eceSum += Math.abs(avgConf - accuracy) * bin.total;
  }
  const ece = total > 0 ? eceSum / total : 0;
  
  // Compute Brier score (average per prediction)
  const brier = total > 0 ? brierSum / total : 0;
  
  return {
    accuracy: total > 0 ? correct / total : 0,
    expectedCalibrationError: ece,
    brierScore: brier,
    perLabelAccuracy: {
      answer: perLabel.answer.total > 0 ? perLabel.answer.correct / perLabel.answer.total : 0,
      refuse: perLabel.refuse.total > 0 ? perLabel.refuse.correct / perLabel.refuse.total : 0,
      escalate: perLabel.escalate.total > 0 ? perLabel.escalate.correct / perLabel.escalate.total : 0
    }
  };
}

/**
 * Compute Wilcoxon signed-rank test comparing accuracy between arms.
 * Uses reliability-eval for validated statistical comparison.
 */
export async function computeWilcoxon(
  items: EvalItem[],
  armAResults: ArmResult[],
  armBResults: ArmResult[]
): Promise<{ pValue: number; significant: boolean; statistic: number; effectSize: number } | undefined> {
  try {
    const { compare } = await import('reliability-eval');
    
    const itemMap = new Map(items.map(item => [item.id, item]));
    
    const pairs: Array<{ id: string; scoreA: number; scoreB: number }> = [];
    
    for (let i = 0; i < armAResults.length; i++) {
      const resultA = armAResults[i];
      const resultB = armBResults[i];
      
      if (!resultA || !resultB || resultA.error || resultB.error) continue;
      if (resultA.itemId !== resultB.itemId) continue;
      
      const item = itemMap.get(resultA.itemId);
      if (!item) continue;
      
      const scoreA = resultA.prediction.label === item.groundTruth ? 1 : 0;
      const scoreB = resultB.prediction.label === item.groundTruth ? 1 : 0;
      
      pairs.push({ id: resultA.itemId, scoreA, scoreB });
    }
    
    if (pairs.length < 2) return undefined;
    
    const mockResultA = {
      items: pairs.map(p => ({
        id: p.id,
        input: '',
        expected: '',
        predicted: '',
        confidence: null,
        correct: p.scoreA === 1,
        score: p.scoreA,
        raw: null
      })),
      metrics: { accuracy: 0, ece: 0, brier: null, n: pairs.length },
      calibrationCurve: [],
      meta: { provider: 'A', model: 'A', startedAt: '', finishedAt: '', durationMs: 0 }
    };
    
    const mockResultB = {
      items: pairs.map(p => ({
        id: p.id,
        input: '',
        expected: '',
        predicted: '',
        confidence: null,
        correct: p.scoreB === 1,
        score: p.scoreB,
        raw: null
      })),
      metrics: { accuracy: 0, ece: 0, brier: null, n: pairs.length },
      calibrationCurve: [],
      meta: { provider: 'B', model: 'B', startedAt: '', finishedAt: '', durationMs: 0 }
    };
    
    const result = compare(mockResultA, mockResultB, { test: 'wilcoxon', metric: 'score' });
    
    return {
      pValue: result.pValue,
      significant: result.significant,
      statistic: result.statistic,
      effectSize: result.effectSize
    };
  } catch (error) {
    console.warn('Wilcoxon test failed:', error);
    return undefined;
  }
}

/**
 * Compute Spearman rank correlation between confidence and correctness.
 * Uses reliability-eval for validated statistical analysis.
 */
export async function computeSpearman(
  items: EvalItem[],
  results: ArmResult[]
): Promise<{ rho: number; pValue: number; significant: boolean } | undefined> {
  try {
    const { correlate } = await import('reliability-eval');
    
    const itemMap = new Map(items.map(item => [item.id, item]));
    
    const validItems: Array<{ id: string; confidence: number; correct: boolean }> = [];
    
    for (const result of results) {
      if (result.error || result.prediction.confidence === undefined) continue;
      
      const item = itemMap.get(result.itemId);
      if (!item) continue;
      
      const correct = result.prediction.label === item.groundTruth;
      
      validItems.push({
        id: result.itemId,
        confidence: result.prediction.confidence,
        correct
      });
    }
    
    if (validItems.length < 2) return undefined;
    
    const mockResult = {
      items: validItems.map(v => ({
        id: v.id,
        input: '',
        expected: '',
        predicted: '',
        confidence: v.confidence,
        correct: v.correct,
        score: v.correct ? 1 : 0,
        raw: null
      })),
      metrics: { accuracy: 0, ece: 0, brier: null, n: validItems.length },
      calibrationCurve: [],
      meta: { provider: '', model: '', startedAt: '', finishedAt: '', durationMs: 0 }
    };
    
    const result = correlate(mockResult, { 
      test: 'spearman', 
      x: 'confidence', 
      y: 'correct' 
    });
    
    return {
      rho: result.rho,
      pValue: result.pValue,
      significant: result.significant
    };
  } catch (error) {
    console.warn('Spearman test failed:', error);
    return undefined;
  }
}

/**
 * Print formatted metrics to console
 */
export function printMetrics(name: string, metrics: Metrics) {
  console.log(`\n=== ${name} ===`);
  console.log(`Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
  console.log(`ECE: ${metrics.expectedCalibrationError.toFixed(4)}`);
  console.log(`Brier Score: ${metrics.brierScore.toFixed(4)}`);
  console.log(`Per-label Accuracy:`);
  console.log(`  answer: ${(metrics.perLabelAccuracy.answer * 100).toFixed(2)}%`);
  console.log(`  refuse: ${(metrics.perLabelAccuracy.refuse * 100).toFixed(2)}%`);
  console.log(`  escalate: ${(metrics.perLabelAccuracy.escalate * 100).toFixed(2)}%`);
}
