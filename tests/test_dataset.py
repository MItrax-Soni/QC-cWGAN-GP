"""
Unit Tests for Facial Expression Dataset Pipeline.
Tests YOLO parser, class mapping, preprocessing, dataset indexing, and DataLoader batching.
"""

import os
import sys
import unittest
import torch
from PIL import Image

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.yolo_parser import (
    CLASS_NAMES,
    class_name_to_id,
    id_to_class_name,
    parse_yolo_line,
    yolo_to_pixel_box
)
from src.data.preprocessing import (
    crop_face_region,
    get_gan_transforms,
    denormalize_tensor
)
from src.data.dataset import FacialExpressionDataset, get_dataloader


class TestDatasetPipeline(unittest.TestCase):

    def test_class_mapping(self):
        """Validates that all 9 classes map bi-directionally without collision."""
        self.assertEqual(len(CLASS_NAMES), 9)
        for idx, name in enumerate(CLASS_NAMES):
            self.assertEqual(class_name_to_id(name), idx)
            self.assertEqual(id_to_class_name(idx), name)

        with self.assertRaises(ValueError):
            class_name_to_id("non_existent_emotion")

        with self.assertRaises(ValueError):
            id_to_class_name(99)

    def test_yolo_parser_and_clamping(self):
        """Tests YOLO string parsing and coordinate clamping within image dimensions."""
        line = "4 0.5 0.5 0.8 0.6"
        parsed = parse_yolo_line(line)
        self.assertIsNotNone(parsed)
        cls_id, xc, yc, w, h = parsed
        self.assertEqual(cls_id, 4)
        self.assertAlmostEqual(xc, 0.5)

        # Convert to 640x640 pixel coordinates
        xmin, ymin, xmax, ymax = yolo_to_pixel_box(xc, yc, w, h, 640, 640, clamp=True)
        self.assertEqual(xmin, 64)
        self.assertEqual(ymin, 128)
        self.assertEqual(xmax, 576)
        self.assertEqual(ymax, 512)

        # Test out-of-bounds coordinates (must clamp gracefully)
        oob_box = yolo_to_pixel_box(1.5, -0.5, 2.0, 2.0, 640, 640, clamp=True)
        self.assertTrue(0 <= oob_box[0] < oob_box[2] <= 640)
        self.assertTrue(0 <= oob_box[1] < oob_box[3] <= 640)

    def test_preprocessing_and_normalization(self):
        """Tests face cropping, 64x64 resizing, and [-1, 1] normalization."""
        dummy_img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        box = (100, 100, 500, 500)
        cropped = crop_face_region(dummy_img, box, margin_ratio=0.0)
        self.assertEqual(cropped.size, (400, 400))

        transform = get_gan_transforms(image_size=64)
        tensor = transform(cropped)

        # Verify tensor attributes
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.all(tensor >= -1.0))
        self.assertTrue(torch.all(tensor <= 1.0))
        self.assertFalse(torch.isnan(tensor).any())

        # Test denormalization
        denorm = denormalize_tensor(tensor)
        self.assertTrue(torch.all(denorm >= 0.0))
        self.assertTrue(torch.all(denorm <= 1.0))

    def test_dataset_and_dataloader(self):
        """Tests end-to-end dataset loading and DataLoader batch generation."""
        manifest_path = r"d:\Projects\GAN\data\splits\pilot_manifest.csv"
        self.assertTrue(os.path.exists(manifest_path), f"Pilot manifest missing: {manifest_path}")

        dataset = FacialExpressionDataset(manifest=manifest_path, image_size=64)
        self.assertEqual(len(dataset), 450)

        # Test single item
        img_tensor, label_tensor = dataset[0]
        self.assertEqual(img_tensor.shape, (3, 64, 64))
        self.assertEqual(label_tensor.dtype, torch.long)
        self.assertTrue(0 <= label_tensor.item() <= 8)

        # Test DataLoader batch
        loader = get_dataloader(manifest=manifest_path, batch_size=16, shuffle=False)
        batch_imgs, batch_lbls = next(iter(loader))

        self.assertEqual(batch_imgs.shape, (16, 3, 64, 64))
        self.assertEqual(batch_lbls.shape, (16,))
        self.assertFalse(torch.isnan(batch_imgs).any())
        self.assertFalse(torch.isinf(batch_imgs).any())


if __name__ == "__main__":
    unittest.main()
