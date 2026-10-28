/**
 * Label -> category taxonomy for in-browser detections (mirrors
 * perception_system/detection/taxonomy.py). Unknown labels map to
 * "unclassified" - the system never invents a category.
 */
import type { Category } from "./types";

const OBJECTS = [
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella", "handbag", "tie",
  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "chair", "couch", "sofa",
  "potted plant", "pottedplant", "bed", "dining table", "diningtable", "toilet", "tv", "tvmonitor", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
];

const MAP: Record<string, Category> = {
  person: "human",
  ...Object.fromEntries(["bicycle", "car", "motorcycle", "motorbike", "airplane", "aeroplane", "bus", "train", "truck", "boat"].map((k) => [k, "vehicle" as Category])),
  ...Object.fromEntries(["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"].map((k) => [k, "animal" as Category])),
  ...Object.fromEntries(["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"].map((k) => [k, "food" as Category])),
  ...Object.fromEntries(OBJECTS.map((k) => [k, "object" as Category])),
};

const ALIASES: Record<string, string> = {
  motorcycle: "bike",
  motorbike: "bike",
  bicycle: "cycle",
  aeroplane: "airplane",
  "cell phone": "phone",
  tv: "television",
  tvmonitor: "television",
  sofa: "couch",
  diningtable: "dining table",
  pottedplant: "potted plant",
};

export function normalizeLabel(label: string): string {
  return label.trim().toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ");
}

export function categoryOf(label: string): Category {
  return MAP[normalizeLabel(label)] ?? "unclassified";
}

export function displayLabel(label: string): string {
  const n = normalizeLabel(label);
  return ALIASES[n] ?? n;
}
