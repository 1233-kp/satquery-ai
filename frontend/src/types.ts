export type InputMode = 'single_image' | 'cross_modal_pair' | 'bi_temporal_pair';

export type ViewName = 'overview' | 'new-analysis' | 'history' | 'capabilities' | 'account';

export type TaskType =
  | 'vqa'
  | 'captioning'
  | 'grounding'
  | 'change_description'
  | 'change_vqa'
  | 'optical_sar_fusion';

export interface UploadResponse {
  image_id: string;
  filename: string;
  modality: string;
  format: string;
  width?: number;
  height?: number;
  bands?: number;
  crs?: string;
}

export interface ExecutionStep {
  step: string;
  detail: string;
}

export interface ExecutionTrace {
  task_selected: TaskType;
  mode: InputMode;
  models_used: string[];
  parameters: Record<string, unknown>;
  steps: ExecutionStep[];
  confidence: number | null;
  confidence_note: string | null;
}

export interface AnalysisResult {
  analysis_id: string;
  query: string;
  answer: string;
  trace: ExecutionTrace;
  visual_evidence: {
    change_mask_png_b64?: string;
    stats?: Record<string, number | string>;
    detection_boxes_png_b64?: string;
    detections?: Array<{ label: string; score: number; box: number[] }>;
  } | null;
  warnings: string[];
  parent_analysis_id?: string | null;
}

export interface AnalysisListItem {
  analysis_id: string;
  query: string;
  task_selected: TaskType;
  created_at: string;
}

export interface ModeConfig {
  mode: InputMode;
  label: string;
  description: string;
  slotLabels: string[];
  examples: string[];
}

export const MODE_CONFIGS: ModeConfig[] = [
  {
    mode: 'single_image',
    label: 'Single image',
    description: 'VQA, captioning, or grounding on one image',
    slotLabels: ['Image'],
    examples: [
      'Describe the land-cover and major objects visible in this image.',
    ],
  },
  {
    mode: 'bi_temporal_pair',
    label: 'Bi-temporal pair',
    description: 'Change detection across two dates',
    slotLabels: ['Before (T1)', 'After (T2)'],
    examples: [
      'What changed between these two dates, and where did the change occur?',
      'Has the built-up area increased, decreased, or remained unchanged?',
    ],
  },
  {
    mode: 'cross_modal_pair',
    label: 'Optical + SAR pair',
    description: 'Joint analysis of co-registered optical and radar images',
    slotLabels: ['Optical', 'SAR'],
    examples: [
      'Use the optical and SAR images together to identify built-up and water-covered regions.',
    ],
  },
];
