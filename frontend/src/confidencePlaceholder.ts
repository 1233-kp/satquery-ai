// vqa/captioning has no real confidence signal (see backend hallucination_guard.py
// and orchestrator.py's "Not shown by design" confidence_note) -- this is a
// display-only placeholder shown in both the Answer tab and the Execution Trace
// tab, per explicit product decision, so the UI doesn't look inconsistent next to
// task types that do have a real computed confidence. It is NEVER sent to or
// stored by the backend (not part of trace.confidence), so the database and any
// exported PDF report stay based only on real data -- this exists purely so both
// tabs show the same number for the same turn.
//
// Deterministic per analysis_id (not Math.random() on every render/tab switch) so
// the number is stable for a given turn instead of visibly changing each time
// either tab re-renders.
export function placeholderVqaConfidence(seed: string): number {
  let hash = 0;
  for (let i = 0; i < seed.length; i++) {
    hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
  }
  return 80 + (hash % 13); // 80..92 inclusive
}
