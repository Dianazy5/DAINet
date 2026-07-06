import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class FineFakeDataset(data.Dataset):
    """
    FineFake adapter for the existing GossipCop-style English multimodal pipeline.

    The original FineFake pickle is a pandas DataFrame with text, image_path,
    topic, fine-grained label, and binary label columns. This dataset keeps the
    original file intact, performs the KEAN binary classification split (6:2:2)
    in memory, and returns the same dictionary keys as FakeNet_dataset:
    content, content_masks, image, clip_image, clip_text, clip_attention_mask,
    label, category.
    """

    TOPIC_TO_ID = {
        "Business": 0,
        "Conflict": 1,
        "Entertainment": 2,
        "Health": 3,
        "Politics": 4,
        "Society": 5,
        "Uncategorized": 6,
    }
    _RECORD_CACHE = {}

    def __init__(
        self,
        root_path: str,
        bert_tokenizer_instance,
        clip_processor_instance,
        split: str = "train",
        image_size: int = 224,
        bert_max_len: int = 197,
        clip_max_len: int = 77,
        seed: int = 2026,
        include_knowledge_text: bool = True,
        max_text_chars: int = 4000,
        max_knowledge_items: int = 8,
        real_label_value: int = 1,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.root_path = root_path
        self.split = split
        self.bert_tokenizer = bert_tokenizer_instance
        self.clip_processor = clip_processor_instance
        self.image_size = image_size
        self.bert_max_len = bert_max_len
        self.clip_max_len = clip_max_len
        self.seed = seed
        self.include_knowledge_text = include_knowledge_text
        self.max_text_chars = max_text_chars
        self.max_knowledge_items = max_knowledge_items
        self.real_label_value = int(real_label_value)

        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"FineFake split '{self.split}' has no usable records.")

        fake_count = sum(1 for item in self.records if item["label"] == 0)
        real_count = sum(1 for item in self.records if item["label"] == 1)
        self.pos_weight = torch.tensor(fake_count / real_count) if real_count else torch.tensor(1.0)
        self.thresh = real_count / len(self.records)

        logger.info(
            "FineFakeDataset split=%s loaded %d records; fake(label=0)=%d, real(label=1)=%d, pos_weight=%.4f",
            self.split,
            len(self.records),
            fake_count,
            real_count,
            float(self.pos_weight),
        )

        self.mae_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_records(self) -> List[Dict[str, Any]]:
        pkl_path = os.path.join(self.root_path, "FineFake.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"FineFake pickle not found: {pkl_path}")

        cache_key = (
            os.path.abspath(self.root_path),
            self.include_knowledge_text,
            self.max_text_chars,
            self.max_knowledge_items,
            self.real_label_value,
        )
        if cache_key in self._RECORD_CACHE:
            usable_records = self._RECORD_CACHE[cache_key]
        else:
            logger.info("Loading FineFake metadata from %s", pkl_path)
            df = pd.read_pickle(pkl_path)
            required_cols = {"text", "image_path", "label", "topic"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"FineFake.pkl missing required columns: {sorted(missing_cols)}")

            usable_records = []
            skipped = 0
            for _, row in df.iterrows():
                try:
                    raw_label = int(row["label"])
                    if raw_label not in (0, 1):
                        skipped += 1
                        continue
                except (TypeError, ValueError):
                    skipped += 1
                    continue

                image_path = self._resolve_image_path(row["image_path"])
                if image_path is None:
                    skipped += 1
                    continue

                topic = str(row.get("topic", "Uncategorized") or "Uncategorized")
                if topic not in self.TOPIC_TO_ID:
                    topic = "Uncategorized"

                content = self._build_content(row)
                label = 1 if raw_label == self.real_label_value else 0
                usable_records.append(
                    {
                        "content": content,
                        "image_path": image_path,
                        "label": label,
                        "raw_label": raw_label,
                        "category": self.TOPIC_TO_ID[topic],
                        "topic": topic,
                        "platform": row.get("platform", ""),
                        "fine_grained_label": row.get("fine-grained label", None),
                    }
                )

            if skipped:
                logger.warning("FineFake skipped %d rows with invalid label or missing image.", skipped)
            self._RECORD_CACHE[cache_key] = usable_records

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(usable_records))
        rng.shuffle(indices)

        train_end = int(len(indices) * 0.6)
        val_end = train_end + int(len(indices) * 0.2)
        split_to_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        selected = split_to_indices[self.split]
        return [usable_records[int(i)] for i in selected]

    def _resolve_image_path(self, image_path_value: Any) -> Optional[str]:
        if image_path_value is None or (isinstance(image_path_value, float) and np.isnan(image_path_value)):
            return None

        image_path = str(image_path_value).strip()
        candidates = []
        if os.path.isabs(image_path):
            candidates.append(image_path)
        candidates.append(os.path.join(self.root_path, image_path))
        candidates.append(os.path.join(self.root_path, image_path.lstrip("./")))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _build_content(self, row: pd.Series) -> str:
        text = self._safe_text(row.get("text", ""))
        parts = [text]
        if self.include_knowledge_text:
            knowledge_text = self._extract_knowledge_text(row.get("description"))
            if knowledge_text:
                parts.append("Knowledge: " + knowledge_text)
        combined = "\n".join(part for part in parts if part)
        return combined[: self.max_text_chars]

    def _extract_knowledge_text(self, description: Any) -> str:
        if description is None or (isinstance(description, float) and np.isnan(description)):
            return ""
        snippets = []
        if isinstance(description, list):
            for item in description:
                if len(snippets) >= self.max_knowledge_items:
                    break
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    snippets.append(str(item[2]))
                elif item is not None:
                    snippets.append(str(item))
        else:
            snippets.append(str(description))
        return "; ".join(s for s in snippets if s)

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image_path = record["image_path"]
        content = record["content"]

        try:
            img_pil = Image.open(image_path).convert("RGB")
            # with Image.open(image_path) as img_raw:
            #     if img_raw.mode == "P" and "transparency" in img_raw.info:
            #         img_raw = img_raw.convert("RGBA")
            #     img_pil = img_raw.convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load FineFake image %s: %s. Using black image.", image_path, exc)
            img_pil = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        try:
            img_mae = self.mae_transform(img_pil)
        except Exception as exc:
            logger.warning("MAE transform failed for %s: %s. Using zeros.", image_path, exc)
            img_mae = torch.zeros((3, self.image_size, self.image_size))

        clip_pixel_values = self._process_clip_image(img_pil)
        bert_input_ids, bert_attention_mask = self._process_bert_text(content)
        clip_input_ids, clip_attention_mask = self._process_clip_text(content)

        return {
            "content": bert_input_ids,
            "content_masks": bert_attention_mask,
            "image": img_mae,
            "clip_image": clip_pixel_values,
            "clip_text": clip_input_ids,
            "clip_attention_mask": clip_attention_mask,
            "label": torch.tensor(record["label"], dtype=torch.float),
            "category": torch.tensor(record["category"], dtype=torch.long),
        }

    def _process_bert_text(self, content: str):
        if self.bert_tokenizer is None:
            return (
                torch.zeros(self.bert_max_len, dtype=torch.long),
                torch.zeros(self.bert_max_len, dtype=torch.long),
            )
        try:
            encoded = self.bert_tokenizer(
                content,
                padding="max_length",
                truncation=True,
                max_length=self.bert_max_len,
                return_tensors="pt",
            )
            return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)
        except Exception as exc:
            logger.warning("FineFake BERT text processing failed: %s. Using zeros.", exc)
            return (
                torch.zeros(self.bert_max_len, dtype=torch.long),
                torch.zeros(self.bert_max_len, dtype=torch.long),
            )

    def _process_clip_image(self, img_pil: Image.Image) -> torch.Tensor:
        default_size = 224
        if self.clip_processor is None:
            return torch.zeros((3, default_size, default_size))
        try:
            processed = self.clip_processor(images=img_pil, return_tensors="pt")
            return processed["pixel_values"].squeeze(0)
        except Exception as exc:
            logger.warning("FineFake CLIP image processing failed: %s. Using zeros.", exc)
            return torch.zeros((3, default_size, default_size))

    def _process_clip_text(self, content: str):
        if self.clip_processor is None:
            return (
                torch.zeros(self.clip_max_len, dtype=torch.long),
                torch.zeros(self.clip_max_len, dtype=torch.long),
            )
        try:
            encoded = self.clip_processor(
                text=content,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.clip_max_len,
            )
            return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)
        except Exception as exc:
            logger.warning("FineFake CLIP text processing failed: %s. Using zeros.", exc)
            return (
                torch.zeros(self.clip_max_len, dtype=torch.long),
                torch.zeros(self.clip_max_len, dtype=torch.long),
            )

    def __len__(self) -> int:
        return len(self.records)
