import type { EvalItem, ArmResult, Prediction, Label } from '../types.js';

/**
 * Arm B: Jev-style choice over three labels with native probabilities
 * Stub implementation that works without API key
 * 
 * ## TypeSafe Jev API Contract (when using live endpoint)
 * 
 * **Request:**
 * POST to JEV_ENDPOINT with Authorization: Bearer <JEV_API_KEY>
 * Content-Type: application/json
 * 
 * Body:
 * {
 *   "question": string,           // The patient/user question
 *   "context": string[],           // Retrieved RAG snippets
 *   "choices": ["answer", "refuse", "escalate"]
 * }
 * 
 * **Response:**
 * Status: 200 OK
 * Content-Type: application/json
 * 
 * Body:
 * {
 *   "probabilities": {
 *     "answer": number,      // [0, 1] - probability to answer
 *     "refuse": number,      // [0, 1] - probability to refuse
 *     "escalate": number     // [0, 1] - probability to escalate
 *   }
 * }
 * 
 * Note: probabilities should sum to ~1.0
 */

interface JevResponse {
  probabilities: {
    answer: number;
    refuse: number;
    escalate: number;
  };
}

async function callJevAPI(question: string, snippets: string[]): Promise<JevResponse> {
  const apiKey = process.env.JEV_API_KEY;
  const endpoint = process.env.JEV_ENDPOINT;
  
  if (!apiKey || !endpoint) {
    throw new Error('JEV_API_KEY or JEV_ENDPOINT not configured');
  }
  
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      question,
      context: snippets,
      choices: ['answer', 'refuse', 'escalate']
    })
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Jev API error: ${response.status} ${response.statusText} - ${errorText}`);
  }
  
  const data = await response.json() as { probabilities: { answer: number; refuse: number; escalate: number } };
  
  const probs = data.probabilities;
  const sum = probs.answer + probs.refuse + probs.escalate;
  if (Math.abs(sum - 1.0) > 0.01) {
    console.warn(`Jev API returned probabilities that sum to ${sum.toFixed(3)}, not 1.0`);
  }
  
  return {
    probabilities: {
      answer: probs.answer,
      refuse: probs.refuse,
      escalate: probs.escalate
    }
  };
}

/**
 * Stub implementation using heuristic-based native probabilities
 * More granular than Arm A to simulate calibrated probability distribution
 */
function stubPredict(item: EvalItem): Prediction {
  const question = item.question.toLowerCase();
  const snippets = item.retrievedSnippets.join(' ').toLowerCase();
  
  // Initialize base probabilities
  let probs = {
    answer: 0.33,
    refuse: 0.33,
    escalate: 0.34
  };
  
  // Evidence quality scoring
  const specificEvidence = [
    'mg', 'dose', 'hours', 'days', 'range', 'score', 'typically',
    'recommended', 'guidelines', 'first-line'
  ];
  const evidenceScore = specificEvidence.filter(kw => snippets.includes(kw)).length;
  
  // Vague/insufficient evidence markers
  const vagueMarkers = [
    'depends on', 'varies', 'individualized', 'multidisciplinary',
    'complex', 'specialized', 'rare'
  ];
  const vagueScore = vagueMarkers.filter(kw => snippets.includes(kw)).length;
  
  // Emergency/urgent markers
  const emergencyMarkers = [
    'emergency', 'immediate', 'urgent', 'life-threatening', 'acute',
    'severe', 'crisis', '911', 'call'
  ];
  const emergencyScore = emergencyMarkers.filter(kw => 
    question.includes(kw) || snippets.includes(kw)
  ).length;
  
  // Out-of-scope markers
  const outOfScope = [
    'stock', 'invest', 'political', 'hack', 'letter', 'legal',
    'diagnose my', 'diagnose me'
  ];
  const oosScore = outOfScope.filter(kw => question.includes(kw)).length;
  
  // Patient-specific decision markers
  const patientSpecific = [
    'should i', 'can i take', 'i\'m taking', 'my mother', 'my child',
    'i have', 'i\'ve been', 'i stopped', 'i started'
  ];
  const psScore = patientSpecific.filter(kw => question.includes(kw)).length;
  
  // Compute probabilities based on features
  if (emergencyScore >= 2 || (emergencyScore >= 1 && psScore >= 1)) {
    // Strong escalate signal
    probs = { answer: 0.05, refuse: 0.10, escalate: 0.85 };
  } else if (emergencyScore === 1 && psScore === 0) {
    // Moderate escalate signal (might be educational about emergencies)
    probs = { answer: 0.30, refuse: 0.15, escalate: 0.55 };
  } else if (psScore >= 2 || (psScore >= 1 && emergencyScore >= 1)) {
    // Patient-specific requiring guidance
    probs = { answer: 0.15, refuse: 0.20, escalate: 0.65 };
  } else if (oosScore >= 1) {
    // Out of scope
    probs = { answer: 0.10, refuse: 0.80, escalate: 0.10 };
  } else if (vagueScore >= 2 && evidenceScore <= 1) {
    // Insufficient evidence
    probs = { answer: 0.20, refuse: 0.70, escalate: 0.10 };
  } else if (evidenceScore >= 3 && vagueScore === 0) {
    // Strong answer signal
    probs = { answer: 0.80, refuse: 0.15, escalate: 0.05 };
  } else if (evidenceScore >= 2) {
    // Moderate answer signal
    probs = { answer: 0.65, refuse: 0.25, escalate: 0.10 };
  } else if (psScore >= 1) {
    // Any patient-specific leans toward escalate
    probs = { answer: 0.25, refuse: 0.30, escalate: 0.45 };
  } else {
    // Default to moderate answer
    probs = { answer: 0.55, refuse: 0.35, escalate: 0.10 };
  }
  
  // Normalize to ensure sum = 1.0
  const sum = probs.answer + probs.refuse + probs.escalate;
  probs.answer /= sum;
  probs.refuse /= sum;
  probs.escalate /= sum;
  
  // Select label with highest probability
  let label: Label = 'answer';
  let maxProb = probs.answer;
  
  if (probs.refuse > maxProb) {
    label = 'refuse';
    maxProb = probs.refuse;
  }
  if (probs.escalate > maxProb) {
    label = 'escalate';
    maxProb = probs.escalate;
  }
  
  return {
    label,
    confidence: maxProb,
    probabilities: probs
  };
}

export async function runJevArm(items: EvalItem[], useStub: boolean = true): Promise<ArmResult[]> {
  const results: ArmResult[] = [];
  
  for (const item of items) {
    const startTime = Date.now();
    
    try {
      let prediction: Prediction;
      
      if (useStub) {
        prediction = stubPredict(item);
      } else {
        const response = await callJevAPI(item.question, item.retrievedSnippets);
        const maxLabel = Object.entries(response.probabilities).reduce((a, b) => 
          b[1] > a[1] ? b : a
        )[0] as Label;
        
        prediction = {
          label: maxLabel,
          confidence: response.probabilities[maxLabel],
          probabilities: response.probabilities
        };
      }
      
      const latencyMs = Date.now() - startTime;
      
      results.push({
        itemId: item.id,
        prediction,
        latencyMs
      });
    } catch (error) {
      results.push({
        itemId: item.id,
        prediction: { 
          label: 'refuse', 
          confidence: 0.33,
          probabilities: { answer: 0.33, refuse: 0.34, escalate: 0.33 }
        },
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
  
  return results;
}
