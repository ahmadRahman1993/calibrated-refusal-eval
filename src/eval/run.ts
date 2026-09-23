import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import * as dotenv from 'dotenv';
import type { EvalItem, EvaluationResults } from '../types.js';
import { runLLMArm } from '../arms/llm.js';
import { runJevArm } from '../arms/jev.js';
import { computeMetrics, computeWilcoxon, computeSpearman, printMetrics } from './metrics.js';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  
  console.log('='.repeat(60));
  console.log('Calibrated Refusal Evaluation Harness');
  console.log('='.repeat(60));
  console.log(`Mode: ${dryRun ? 'DRY RUN (deterministic predictions)' : 'LIVE API CALLS'}`);
  console.log();
  
  // Load seed data
  const seedPath = path.join(__dirname, '../data/seed.json');
  const seedData = JSON.parse(fs.readFileSync(seedPath, 'utf-8')) as EvalItem[];
  
  console.log(`Loaded ${seedData.length} evaluation items`);
  console.log();
  
  // Distribution summary
  const distribution = seedData.reduce((acc, item) => {
    acc[item.groundTruth] = (acc[item.groundTruth] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  console.log('Ground truth distribution:');
  Object.entries(distribution).forEach(([label, count]) => {
    console.log(`  ${label}: ${count} (${((count / seedData.length) * 100).toFixed(1)}%)`);
  });
  console.log();
  
  // Run Arm A (LLM with elicited confidence)
  console.log('Running Arm A: Prompted LLM with elicited confidence...');
  const armAResults = await runLLMArm(seedData, dryRun);
  const armAMetrics = computeMetrics(seedData, armAResults);
  printMetrics('Arm A: Prompted LLM', armAMetrics);
  
  // Run Arm B (Jev-style native probabilities)
  console.log('\nRunning Arm B: Jev-style native probabilities...');
  const armBResults = await runJevArm(seedData, dryRun);
  const armBMetrics = computeMetrics(seedData, armBResults);
  printMetrics('Arm B: Jev-style native probabilities', armBMetrics);
  
  // Comparison
  console.log('\n=== Comparison ===');
  console.log(`Accuracy difference: ${((armBMetrics.accuracy - armAMetrics.accuracy) * 100).toFixed(2)}% (B - A)`);
  console.log(`ECE difference: ${(armBMetrics.expectedCalibrationError - armAMetrics.expectedCalibrationError).toFixed(4)} (B - A)`);
  console.log(`Brier difference: ${(armBMetrics.brierScore - armAMetrics.brierScore).toFixed(4)} (B - A)`);
  
  const wilcoxonResult = await computeWilcoxon(seedData, armAResults, armBResults);
  const spearmanAResult = await computeSpearman(seedData, armAResults);
  const spearmanBResult = await computeSpearman(seedData, armBResults);
  
  console.log(`\nWilcoxon signed-rank test (A vs B):`);
  if (wilcoxonResult) {
    console.log(`  p-value: ${wilcoxonResult.pValue.toFixed(4)}`);
    console.log(`  significant (α=0.05): ${wilcoxonResult.significant ? 'Yes' : 'No'}`);
    console.log(`  effect size: ${wilcoxonResult.effectSize.toFixed(4)}`);
  } else {
    console.log(`  not computed (insufficient data)`);
  }
  
  console.log(`\nSpearman ρ (confidence ↔ correctness):`);
  console.log(`  Arm A: ${spearmanAResult ? `ρ=${spearmanAResult.rho.toFixed(4)}, p=${spearmanAResult.pValue.toFixed(4)}` : 'not computed'}`);
  console.log(`  Arm B: ${spearmanBResult ? `ρ=${spearmanBResult.rho.toFixed(4)}, p=${spearmanBResult.pValue.toFixed(4)}` : 'not computed'}`);
  
  // Notes
  const notes: string[] = [];
  if (dryRun) {
    notes.push('Dry-run mode: predictions are deterministic heuristics, not real API calls');
  }
  if (!wilcoxonResult) {
    notes.push('Wilcoxon test requires at least 2 paired observations');
  }
  if (!spearmanAResult && !spearmanBResult) {
    notes.push('Spearman correlation requires confidence values and at least 2 observations');
  }
  
  if (notes.length > 0) {
    console.log('\nNotes:');
    notes.forEach(note => console.log(`  - ${note}`));
  }
  
  // Save results
  const results: EvaluationResults = {
    timestamp: new Date().toISOString(),
    totalItems: seedData.length,
    armA: {
      name: 'Prompted LLM',
      results: armAResults,
      metrics: armAMetrics
    },
    armB: {
      name: 'Jev-style native probabilities',
      results: armBResults,
      metrics: armBMetrics
    },
    comparison: {
      wilcoxonP: wilcoxonResult?.pValue,
      wilcoxonSignificant: wilcoxonResult?.significant,
      wilcoxonEffectSize: wilcoxonResult?.effectSize,
      spearmanArmA: spearmanAResult ? {
        rho: spearmanAResult.rho,
        pValue: spearmanAResult.pValue,
        significant: spearmanAResult.significant
      } : undefined,
      spearmanArmB: spearmanBResult ? {
        rho: spearmanBResult.rho,
        pValue: spearmanBResult.pValue,
        significant: spearmanBResult.significant
      } : undefined,
      notes
    }
  };
  
  const outputPath = path.join(__dirname, '../../results.json');
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
  console.log(`\nResults saved to: ${outputPath}`);
  
  console.log('\n' + '='.repeat(60));
  console.log('Evaluation complete!');
  console.log('='.repeat(60));
}

main().catch(error => {
  console.error('Error:', error);
  process.exit(1);
});
