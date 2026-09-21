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
  const armBResults = await runJevArm(seedData, true); // Always use stub for now
  const armBMetrics = computeMetrics(seedData, armBResults);
  printMetrics('Arm B: Jev-style (stub)', armBMetrics);
  
  // Comparison
  console.log('\n=== Comparison ===');
  console.log(`Accuracy difference: ${((armBMetrics.accuracy - armAMetrics.accuracy) * 100).toFixed(2)}% (B - A)`);
  console.log(`ECE difference: ${(armBMetrics.expectedCalibrationError - armAMetrics.expectedCalibrationError).toFixed(4)} (B - A)`);
  console.log(`Brier difference: ${(armBMetrics.brierScore - armAMetrics.brierScore).toFixed(4)} (B - A)`);
  
  const wilcoxonP = computeWilcoxon(armAResults, armBResults);
  const spearmanRho = computeSpearman(armAResults, armBResults);
  
  console.log(`\nWilcoxon p-value: ${wilcoxonP !== undefined ? wilcoxonP.toFixed(4) : 'not computed (needs reliability-eval integration)'}`);
  console.log(`Spearman ρ: ${spearmanRho !== undefined ? spearmanRho.toFixed(4) : 'not computed (needs reliability-eval integration)'}`);
  
  // Notes
  const notes: string[] = [];
  if (dryRun) {
    notes.push('Dry-run mode: predictions are deterministic heuristics, not real API calls');
  }
  if (wilcoxonP === undefined) {
    notes.push('Wilcoxon test requires reliability-eval integration (github.com/ahmadRahman1993/reliability-eval)');
  }
  if (spearmanRho === undefined) {
    notes.push('Spearman correlation requires reliability-eval integration');
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
      wilcoxonP,
      spearmanRho,
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
