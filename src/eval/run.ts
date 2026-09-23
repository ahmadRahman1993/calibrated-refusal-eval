import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import * as dotenv from 'dotenv';
import type { EvalItem, EvaluationResults, ArmResult, PairwiseComparison } from '../types.js';
import { runLLMArm } from '../arms/llm.js';
import { runJevArm } from '../arms/jev.js';
import { computeMetrics, computeWilcoxon, computeSpearman, printMetrics } from './metrics.js';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Determine which arms are available based on environment variables
 */
function getAvailableArms(dryRun: boolean): { openai: boolean; gemini: boolean; jev: boolean } {
  if (dryRun) {
    // Dry-run: always enable all three arms with stubs
    return { openai: true, gemini: true, jev: true };
  }
  
  // Live mode: check for API keys
  const openai = !!process.env.OPENAI_API_KEY;
  const gemini = !!process.env.GEMINI_API_KEY;
  const jev = !!(process.env.AI_GATEWAY_API_KEY || (process.env.JEV_API_KEY && process.env.JEV_ENDPOINT));
  
  return { openai, gemini, jev };
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  
  console.log('='.repeat(60));
  console.log('Calibrated Refusal Evaluation Harness (3-Way)');
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
  
  // Determine available arms
  const availableArms = getAvailableArms(dryRun);
  
  if (!dryRun) {
    const enabledCount = Object.values(availableArms).filter(Boolean).length;
    
    if (enabledCount < 2) {
      console.error('\n❌ ERROR: Live mode requires at least 2 arms with API keys configured.');
      console.error('\nMissing API keys:');
      if (!availableArms.openai) console.error('  - OPENAI_API_KEY not set');
      if (!availableArms.gemini) console.error('  - GEMINI_API_KEY not set');
      if (!availableArms.jev) console.error('  - AI_GATEWAY_API_KEY (or JEV_API_KEY + JEV_ENDPOINT) not set');
      console.error('\nSet at least 2 of these keys in .env to run a live evaluation.');
      console.error('Run "npm run eval:dry" for a dry-run with all three arms (no keys required).\n');
      process.exit(1);
    }
    
    console.log('Available arms:');
    if (availableArms.openai) console.log('  ✓ OpenAI');
    if (availableArms.gemini) console.log('  ✓ Gemini');
    if (availableArms.jev) console.log('  ✓ Jev');
    console.log();
  } else {
    console.log('Dry-run mode: running all three arms with deterministic stubs\n');
  }
  
  // Run arms
  const armResults: Record<string, ArmResult[]> = {};
  
  // OpenAI arm
  if (availableArms.openai) {
    console.log('Running OpenAI arm: GPT-4 with elicited confidence...');
    armResults.openai = await runLLMArm(seedData, dryRun, 'openai');
    const metrics = computeMetrics(seedData, armResults.openai);
    printMetrics('OpenAI', metrics);
  }
  
  // Gemini arm
  if (availableArms.gemini) {
    console.log('\nRunning Gemini arm: Gemini 1.5 with elicited confidence...');
    armResults.gemini = await runLLMArm(seedData, dryRun, 'gemini');
    const metrics = computeMetrics(seedData, armResults.gemini);
    printMetrics('Gemini', metrics);
  }
  
  // Jev arm
  if (availableArms.jev) {
    console.log('\nRunning Jev arm: native probabilities...');
    armResults.jev = await runJevArm(seedData, dryRun);
    const metrics = computeMetrics(seedData, armResults.jev);
    printMetrics('Jev', metrics);
  }
  
  // Compute Spearman correlations
  console.log('\n=== Spearman ρ (confidence ↔ correctness) ===');
  const spearmanResults: Record<string, { rho: number; pValue: number; significant: boolean }> = {};
  
  for (const [armName, results] of Object.entries(armResults)) {
    const spearman = await computeSpearman(seedData, results);
    if (spearman) {
      spearmanResults[armName] = spearman;
      console.log(`  ${armName}: ρ=${spearman.rho.toFixed(4)}, p=${spearman.pValue.toFixed(4)} ${spearman.significant ? '✓' : ''}`);
    } else {
      console.log(`  ${armName}: not computed (insufficient data)`);
    }
  }
  
  // Compute pairwise comparisons (all pairs)
  console.log('\n=== Pairwise Comparisons (Wilcoxon signed-rank test) ===');
  const pairwiseComparisons: PairwiseComparison[] = [];
  const armNames = Object.keys(armResults);
  
  for (let i = 0; i < armNames.length; i++) {
    for (let j = i + 1; j < armNames.length; j++) {
      const armA = armNames[i];
      const armB = armNames[j];
      
      const wilcoxon = await computeWilcoxon(seedData, armResults[armA], armResults[armB]);
      
      console.log(`\n${armA} vs ${armB}:`);
      if (wilcoxon) {
        console.log(`  p-value: ${wilcoxon.pValue.toFixed(4)}`);
        console.log(`  significant (α=0.05): ${wilcoxon.significant ? 'Yes ✓' : 'No'}`);
        console.log(`  effect size: ${wilcoxon.effectSize.toFixed(4)}`);
        
        pairwiseComparisons.push({
          armA,
          armB,
          wilcoxonP: wilcoxon.pValue,
          wilcoxonSignificant: wilcoxon.significant,
          wilcoxonEffectSize: wilcoxon.effectSize
        });
      } else {
        console.log(`  not computed (insufficient data)`);
        pairwiseComparisons.push({ armA, armB });
      }
    }
  }
  
  // Accuracy comparison table
  console.log('\n=== Accuracy Summary ===');
  const accuracies: Record<string, number> = {};
  for (const [armName, results] of Object.entries(armResults)) {
    const metrics = computeMetrics(seedData, results);
    accuracies[armName] = metrics.accuracy;
    console.log(`  ${armName}: ${(metrics.accuracy * 100).toFixed(2)}%`);
  }
  
  // Notes
  const notes: string[] = [];
  if (dryRun) {
    notes.push('Dry-run mode: predictions are deterministic heuristics, not real API calls');
  }
  if (Object.keys(spearmanResults).length === 0) {
    notes.push('Spearman correlation requires confidence values and at least 2 observations');
  }
  if (pairwiseComparisons.every(c => c.wilcoxonP === undefined)) {
    notes.push('Wilcoxon test requires at least 2 paired observations');
  }
  
  if (notes.length > 0) {
    console.log('\nNotes:');
    notes.forEach(note => console.log(`  - ${note}`));
  }
  
  // Save results
  const arms: Record<string, { name: string; results: ArmResult[]; metrics: ReturnType<typeof computeMetrics> }> = {};
  for (const [armName, results] of Object.entries(armResults)) {
    arms[armName] = {
      name: armName,
      results,
      metrics: computeMetrics(seedData, results)
    };
  }
  
  const results: EvaluationResults = {
    timestamp: new Date().toISOString(),
    totalItems: seedData.length,
    arms,
    spearman: spearmanResults,
    pairwiseComparisons,
    notes
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
