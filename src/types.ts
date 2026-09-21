export type Label = 'answer' | 'refuse' | 'escalate';

/**
 * A single evaluation item containing a medical question and retrieved RAG snippets.
 */
export interface EvalItem {
  id: string;
  question: string;
  retrievedSnippets: string[];
  groundTruth: Label;
  notes?: string;
}

/**
 * Prediction from an arm with label and confidence/probability.
 */
export interface Prediction {
  label: Label;
  confidence: number; // 0-1
  probabilities?: {
    answer: number;
    refuse: number;
    escalate: number;
  };
}

/**
 * Arm prediction result including metadata.
 */
export interface ArmResult {
  itemId: string;
  prediction: Prediction;
  latencyMs?: number;
  error?: string;
}

/**
 * Evaluation metrics for an arm.
 */
export interface Metrics {
  accuracy: number;
  expectedCalibrationError: number;
  brierScore: number;
  perLabelAccuracy: {
    answer: number;
    refuse: number;
    escalate: number;
  };
}

/**
 * Complete evaluation results comparing arms.
 */
export interface EvaluationResults {
  timestamp: string;
  totalItems: number;
  armA: {
    name: string;
    results: ArmResult[];
    metrics: Metrics;
  };
  armB: {
    name: string;
    results: ArmResult[];
    metrics: Metrics;
  };
  comparison: {
    wilcoxonP?: number;
    spearmanRho?: number;
    notes: string[];
  };
}
