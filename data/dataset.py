"""
Dataset classes for Multimodal Disease Classification

Handles CT images (COVID, Kidney, Lungs DICOM) and lab values together.
Supports both synthetic and real NHANES blood report data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import xml.etree.ElementTree as ET
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    IMAGE_SIZE, MEAN, STD, CLASS_NAMES, 
    TRAIN_RATIO, VAL_RATIO, SEED, BATCH_SIZE
)
from data.lab_generator import LabValueGenerator, LabNormalizer


class NHANESLabLoader:
    """
    Load and categorize real blood report data from NHANES XPT files.
    Assigns lab values to disease categories based on medical criteria.
    """
    
    def __init__(self, dataset_root, seed=SEED):
        """
        Args:
            dataset_root: Path to dataset folder containing CBC_J.xpt and HSCRP_J.xpt
        """
        self.dataset_root = dataset_root
        self.seed = seed
        np.random.seed(seed)
        
        # Load and merge NHANES data
        self.lab_data = self._load_nhanes_data()
        
        # Categorize samples by disease markers
        self.categorized_data = self._categorize_by_disease()
        
        # Track assignment indices
        self.assignment_indices = {0: 0, 1: 0, 2: 0, 3: 0}
        
    def _load_nhanes_data(self):
        """Load CBC and CRP data from XPT files"""
        cbc_path = os.path.join(self.dataset_root, 'CBC_J.xpt')
        crp_path = os.path.join(self.dataset_root, 'HSCRP_J.xpt')
        
        if not os.path.exists(cbc_path) or not os.path.exists(crp_path):
            print(f"Warning: NHANES files not found at {self.dataset_root}")
            return None
        
        # Load data
        cbc = pd.read_sas(cbc_path, format='xport')
        crp = pd.read_sas(crp_path, format='xport')
        
        # Merge on SEQN (subject ID)
        merged = pd.merge(
            cbc[['SEQN', 'LBXWBCSI', 'LBXHGB']], 
            crp[['SEQN', 'LBXHSCRP']], 
            on='SEQN'
        )
        
        # Drop rows with missing values
        merged = merged.dropna().reset_index(drop=True)
        
        # Rename columns for clarity
        merged = merged.rename(columns={
            'LBXWBCSI': 'WBC',  # WBC in 10^9/L
            'LBXHGB': 'Hemoglobin',  # Hemoglobin in g/dL
            'LBXHSCRP': 'CRP'  # CRP in mg/L
        })
        
        print(f"Loaded {len(merged)} real NHANES blood samples")
        return merged
    
    def _categorize_by_disease(self):
        """
        Categorize lab samples into disease classes based on medical criteria:
        
        Normal (0): All values within normal range
        - WBC: 4.0-10.0 (10^9/L)
        - Hemoglobin: 12.0-17.5 g/dL
        - CRP: <3.0 mg/L
        
        Tumor (1): Possible anemia with normal/mild inflammation
        - Low hemoglobin (<12 g/dL) with mild CRP elevation (1-10 mg/L)
        
        Infection (2): High WBC and elevated CRP (inflammatory markers)
        - WBC >10.0 (10^9/L) OR CRP >10 mg/L
        
        Inflammatory (3): Elevated CRP with normal/low WBC
        - CRP 3-10 mg/L with WBC <10
        """
        if self.lab_data is None:
            return {0: [], 1: [], 2: [], 3: []}
        
        categorized = {0: [], 1: [], 2: [], 3: []}
        
        for idx, row in self.lab_data.iterrows():
            wbc = row['WBC']
            hgb = row['Hemoglobin']
            crp = row['CRP']
            
            lab_values = np.array([crp, wbc, hgb], dtype=np.float32)
            
            # Infection: High inflammatory markers
            if crp > 10 or wbc > 12:
                categorized[2].append(lab_values)
            # Tumor: Low hemoglobin with mild inflammation
            elif hgb < 11.5 and crp >= 1 and crp <= 10:
                categorized[1].append(lab_values)
            # Inflammatory: Moderate CRP elevation
            elif crp >= 3 and crp <= 10 and wbc <= 10:
                categorized[3].append(lab_values)
            # Normal: All values in normal range
            elif 4 <= wbc <= 10 and 12 <= hgb <= 17.5 and crp < 3:
                categorized[0].append(lab_values)
        
        # Shuffle each category
        for class_id in categorized:
            np.random.shuffle(categorized[class_id])
        
        print("NHANES lab data categorization:")
        for class_id, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {len(categorized[class_id])} samples")
        
        return categorized
    
    def get_lab_for_class(self, class_id):
        """
        Get a real lab value for a given disease class.
        Cycles through available samples.
        
        Args:
            class_id: Disease class (0-3)
        
        Returns:
            numpy array [CRP, WBC, Hemoglobin]
        """
        samples = self.categorized_data[class_id]
        
        if len(samples) == 0:
            # Fallback to synthetic if no real samples
            return None
        
        idx = self.assignment_indices[class_id] % len(samples)
        self.assignment_indices[class_id] += 1
        
        return samples[idx].copy()
    
    def get_normalization_stats(self):
        """Get mean and std from real data for normalization"""
        if self.lab_data is None:
            return None
        
        # Order: CRP, WBC, Hemoglobin
        mean = np.array([
            self.lab_data['CRP'].mean(),
            self.lab_data['WBC'].mean(),
            self.lab_data['Hemoglobin'].mean()
        ])
        std = np.array([
            self.lab_data['CRP'].std(),
            self.lab_data['WBC'].std(),
            self.lab_data['Hemoglobin'].std()
        ])
        
        return {'mean': mean, 'std': std}

# Optional import for DICOM
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM datasets will not work.")

from torchvision.transforms import functional as TF

# --- LIDC-IDRI DICOM+XML Dataset Loader ---
class LIDCIDRIDicomDataset(Dataset):
    """
    Dataset for LIDC-IDRI lung CT scans with nodule/non-nodule labels.
    Expects:
        root_dirs: list of root folders (nodules and no_nodules)
        transform: torchvision transforms for images
        label_map: dict, e.g. {'nodule': 1, 'non-nodule': 0}
    Returns:
        dict with 'image', 'label', 'path', 'meta'
    """
    def __init__(self, root_dirs, transform=None, label_map=None):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.label_map = label_map or {'nodule': 1, 'non-nodule': 0}
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for root_dir in self.root_dirs:
            for patient in os.listdir(root_dir):
                patient_path = os.path.join(root_dir, patient)
                if not os.path.isdir(patient_path):
                    continue
                for study in os.listdir(patient_path):
                    study_path = os.path.join(patient_path, study)
                    if not os.path.isdir(study_path):
                        continue
                    for series in os.listdir(study_path):
                        series_path = os.path.join(study_path, series)
                        if not os.path.isdir(series_path):
                            continue
                        # Find XML annotation file
                        xml_file = None
                        for f in os.listdir(series_path):
                            if f.lower().endswith('.xml'):
                                xml_file = os.path.join(series_path, f)
                                break
                        if not xml_file:
                            continue
                        # Parse XML for findings
                        findings = LIDCIDRIAnnotationLoader('').parse_xml(xml_file)
                        # Map imageSOP_UID to label
                        sop_to_label = {}
                        for ann in findings:
                            if 'imageSOP_UID' in ann and ann['imageSOP_UID']:
                                label = self.label_map.get(ann['type'], -1)
                                sop_to_label[ann['imageSOP_UID']] = label
                        # Find DICOMs and match to label
                        for f in os.listdir(series_path):
                            if f.lower().endswith('.dcm'):
                                dcm_path = os.path.join(series_path, f)
                                try:
                                    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                                    sop_uid = str(dcm.SOPInstanceUID)
                                    if sop_uid in sop_to_label:
                                        samples.append({
                                            'dcm_path': dcm_path,
                                            'label': sop_to_label[sop_uid],
                                            'meta': {'patient': patient, 'study': study, 'series': series, 'sop_uid': sop_uid, 'xml': xml_file}
                                        })
                                except Exception as e:
                                    print(f"DICOM read error: {dcm_path}: {e}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dcm = pydicom.dcmread(sample['dcm_path'])
        img = dcm.pixel_array.astype('float32')
        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return {
            'image': img,
            'label': label,
            'path': sample['dcm_path'],
            'meta': sample['meta']
        }
import xml.etree.ElementTree as ET

# --- LIDC-IDRI XML Annotation Loader ---
class LIDCIDRIAnnotationLoader:
    """
    Loader for LIDC-IDRI XML annotation files.
    Extracts nodule and non-nodule findings per scan.
    Returns a list of dicts with annotation info.
    """
    def __init__(self, xml_dir):
        self.xml_dir = xml_dir
        self.annotation_files = self._find_xml_files()

    def _find_xml_files(self):
        xml_files = []
        for root, _, files in os.walk(self.xml_dir):
            for f in files:
                if f.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root, f))
        return xml_files

    def parse_all(self):
        """Parse all XML files and return a list of findings."""
        all_findings = []
        for xml_path in self.annotation_files:
            findings = self.parse_xml(xml_path)
            all_findings.extend(findings)
        return all_findings

    def parse_xml(self, xml_path):
        """Parse a single XML file and extract findings."""
        findings = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {'lidc': root.tag.split('}')[0].strip('{')}
            # Get scan-level UIDs
            header = root.find('lidc:ResponseHeader', ns)
            study_uid = header.findtext('lidc:StudyInstanceUID', default='', namespaces=ns)
            series_uid = header.findtext('lidc:SeriesInstanceUid', default='', namespaces=ns)
            # Parse reading sessions
            for session in root.findall('lidc:readingSession', ns):
                # Non-nodules
                for nn in session.findall('lidc:nonNodule', ns):
                    finding = {
                        'type': 'non-nodule',
                        'nonNoduleID': nn.findtext('lidc:nonNoduleID', default='', namespaces=ns),
                        'imageSOP_UID': nn.findtext('lidc:imageSOP_UID', default='', namespaces=ns),
                        'z': nn.findtext('lidc:imageZposition', default='', namespaces=ns),
                        'x': nn.find('lidc:locus/lidc:xCoord', ns).text if nn.find('lidc:locus/lidc:xCoord', ns) is not None else '',
                        'y': nn.find('lidc:locus/lidc:yCoord', ns).text if nn.find('lidc:locus/lidc:yCoord', ns) is not None else '',
                        'study_uid': study_uid,
                        'series_uid': series_uid,
                        'xml_path': xml_path
                    }
                    findings.append(finding)
                # Nodules
                for nodule in session.findall('lidc:unblindedReadNodule', ns):
                    nodule_id = nodule.findtext('lidc:noduleID', default='', namespaces=ns)
                    for roi in nodule.findall('lidc:roi', ns):
                        finding = {
                            'type': 'nodule',
                            'noduleID': nodule_id,
                            'imageSOP_UID': roi.findtext('lidc:imageSOP_UID', default='', namespaces=ns),
                            'z': roi.findtext('lidc:imageZposition', default='', namespaces=ns),
                            'inclusion': roi.findtext('lidc:inclusion', default='', namespaces=ns),
                            'edge_coords': [
                                (em.findtext('lidc:xCoord', default='', namespaces=ns),
                                 em.findtext('lidc:yCoord', default='', namespaces=ns))
                                for em in roi.findall('lidc:edgeMap', ns)
                            ],
                            'study_uid': study_uid,
                            'series_uid': series_uid,
                            'xml_path': xml_path
                        }
                        findings.append(finding)
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
        return findings
class COVIDCTClassFolderDataset(Dataset):
    """
    Dataset for COVID CT images organized in class folders (COVID, non-COVID).
    Expects:
        root_dir: path to ctscan_covid (should contain COVID/ and non-COVID/)
        class_map: dict mapping folder names to class indices (default: {'COVID': 1, 'non-COVID': 0})
        lab_normalizer: LabNormalizer instance for z-score normalization
        transform: Torchvision transforms for images
        augment: Whether to apply data augmentation
    """
    def __init__(self, root_dir, class_map=None, lab_normalizer=None, transform=None, augment=False):
        self.root_dir = root_dir
        self.class_map = class_map or {'COVID': 1, 'non-COVID': 0}
        self.lab_normalizer = lab_normalizer
        self.augment = augment
        self.image_paths = []
        self.labels = []
        for folder, class_idx in self.class_map.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist")
                continue
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(class_idx)
        # Lab value generator (synthetic, since no real labs)
        self.lab_generator = LabValueGenerator(seed=SEED)
        self.lab_values = self._generate_lab_values()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

    def _generate_lab_values(self):
        # For COVID: use Infection class stats; for non-COVID: use Normal class stats
        labs = []
        for label in self.labels:
            # 1 = COVID (use Infection), 0 = non-COVID (use Normal)
            class_id = 2 if label == 1 else 0
            lab = self.lab_generator.generate_for_class_id(class_id, n_samples=1)[0]
            labs.append(lab)
        labs = np.array(labs)
        if self.lab_normalizer is not None:
            labs = self.lab_normalizer.transform(labs)
        return labs

    def _get_default_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[MEAN], std=[STD])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[MEAN], std=[STD])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        lab = torch.tensor(self.lab_values[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            'image': image,
            'lab': lab,
            'label': label,
            'path': img_path
        }

def create_covid_ct_classfolder_loaders(base_dir, batch_size=BATCH_SIZE, transform=None, augment=False):
    """
    Utility to create DataLoader for COVID CT dataset (class-folder style).
    Args:
        base_dir: path to ctscan_covid (should contain COVID/ and non-COVID/)
        batch_size: batch size
        transform: torchvision transforms
        augment: whether to use augmentation (for train)
    Returns:
        train_loader, val_loader, test_loader, lab_normalizer
    """
    # Lab normalizer (use Infection and Normal stats only)
    lab_gen = LabValueGenerator()
    stats = lab_gen.get_normalization_stats()
    lab_normalizer = LabNormalizer(mean=stats['mean'], std=stats['std'])

    # Gather all image paths and labels
    dataset = COVIDCTClassFolderDataset(base_dir, lab_normalizer=lab_normalizer, augment=augment, transform=transform)
    total_size = len(dataset)
    indices = np.arange(total_size)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    train_end = int(TRAIN_RATIO * total_size)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total_size)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"COVID CT Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, lab_normalizer
class YOLOCTDataset(Dataset):
    """
    Dataset for YOLO-style CT images and bounding box labels.
    Expects:
        images_dir: path to images (e.g., .../train/images)
        labels_dir: path to labels (e.g., .../train/labels)
        Each image has a corresponding .txt file with YOLO format:
            class x_center y_center width height (all normalized)
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # Find corresponding label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        # Load image

        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Ensure image shape is [1, H, W] (no extra channel or batch dim)
        if image.dim() == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] != 1:
            # If shape is [H, W, 1], permute to [1, H, W]
            image = image.permute(2, 0, 1)
        if image.dim() == 2:
            image = image.unsqueeze(0)

        # Load labels (YOLO format)
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append([x, y, w, h])
                        classes.append(int(cls))
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)

        return {
            'image': image,
            'boxes': boxes,  # (N, 4) normalized
            'labels': classes,  # (N,)
            'path': img_path
        }

def create_yolo_ct_loaders(base_dir, split='train', batch_size=BATCH_SIZE, transform=None):
    """
    Utility to create DataLoader for YOLO-style kidney CT dataset.
    Args:
        base_dir: path to ctscans_kidney (should contain train/valid/test)
        split: 'train', 'valid', or 'test'
    Returns:
        DataLoader for the split
    """
    images_dir = os.path.join(base_dir, split, 'images')
    labels_dir = os.path.join(base_dir, split, 'labels')
    dataset = YOLOCTDataset(images_dir, labels_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0)
    return loader


class CTLabDataset(Dataset):
    """
    Dataset for CT images paired with lab values.
    
    Expected folder structure:
    data/
    ├── Normal/
    │   ├── img001.png
    │   ├── img002.png
    │   └── ...
    ├── Tumor/
    │   └── ...
    ├── Infection/
    │   └── ...
    └── Inflammatory/
        └── ...
    
    Lab values are synthetically generated based on the class.
    """
    
    def __init__(self, image_paths, labels, lab_normalizer=None, 
                 transform=None, augment=False):
        """
        Args:
            image_paths: List of paths to CT images
            labels: List of class labels (0-3)
            lab_normalizer: LabNormalizer instance for z-score normalization
            transform: Torchvision transforms for images
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.lab_normalizer = lab_normalizer
        self.augment = augment
        
        # Initialize lab value generator
        self.lab_generator = LabValueGenerator(seed=SEED)
        
        # Pre-generate lab values for consistency
        self.lab_values = self._generate_lab_values()
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def _generate_lab_values(self):
        """Generate synthetic lab values for all samples."""
        labs = []
        for label in self.labels:
            lab = self.lab_generator.generate_for_class_id(label, n_samples=1)[0]
            labs.append(lab)
        labs = np.array(labs)
        
        # Normalize if normalizer provided
        if self.lab_normalizer is not None:
            labs = self.lab_normalizer.transform(labs)
        
        return labs
    
    def _get_default_transform(self):
        """Default image transforms."""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[MEAN], std=[STD])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[MEAN], std=[STD])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Get lab values
        lab = torch.tensor(self.lab_values[idx], dtype=torch.float32)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'lab': lab,
            'label': label,
            'path': img_path
        }


class SyntheticCTDataset(Dataset):
    """
    Synthetic dataset for testing when real CT images are not available.
    Generates random noise images with class-specific patterns.
    """
    
    def __init__(self, n_samples_per_class=100, lab_normalizer=None, 
                 augment=False, seed=SEED):
        """
        Args:
            n_samples_per_class: Number of synthetic samples per class
            lab_normalizer: LabNormalizer instance
            augment: Whether to apply augmentation
            seed: Random seed
        """
        self.n_samples_per_class = n_samples_per_class
        self.n_classes = len(CLASS_NAMES)
        self.total_samples = n_samples_per_class * self.n_classes
        self.lab_normalizer = lab_normalizer
        self.augment = augment
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate labels
        self.labels = []
        for class_id in range(self.n_classes):
            self.labels.extend([class_id] * n_samples_per_class)
        self.labels = np.array(self.labels)
        
        # Generate synthetic images and labs
        self.images = self._generate_synthetic_images()
        self.lab_generator = LabValueGenerator(seed=seed)
        self.lab_values = self._generate_lab_values()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
    
    def _generate_synthetic_images(self):
        """Generate synthetic CT-like images with class-specific patterns."""
        images = []
        
        for label in self.labels:
            # Base noise
            img = np.random.randn(IMAGE_SIZE, IMAGE_SIZE) * 0.1
            
            # Add class-specific patterns
            if label == 0:  # Normal - relatively uniform
                img += 0.5
            elif label == 1:  # Tumor - bright spot
                cx, cy = np.random.randint(50, 174, 2)
                r = np.random.randint(15, 40)
                y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                img[mask] += 0.4
                img += 0.3
            elif label == 2:  # Infection - scattered bright regions
                for _ in range(np.random.randint(3, 8)):
                    cx, cy = np.random.randint(30, 194, 2)
                    r = np.random.randint(10, 25)
                    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                    mask = (x - cx)**2 + (y - cy)**2 <= r**2
                    img[mask] += 0.3
                img += 0.35
            elif label == 3:  # Inflammatory - diffuse pattern
                img += 0.4
                # Add some texture
                texture = np.random.randn(IMAGE_SIZE, IMAGE_SIZE) * 0.15
                img += texture
            
            # Clip to valid range
            img = np.clip(img, 0, 1)
            images.append(img)
        
        return np.array(images, dtype=np.float32)
    
    def _generate_lab_values(self):
        """Generate synthetic lab values for all samples."""
        labs = []
        for label in self.labels:
            lab = self.lab_generator.generate_for_class_id(label, n_samples=1)[0]
            labs.append(lab)
        labs = np.array(labs)
        
        if self.lab_normalizer is not None:
            labs = self.lab_normalizer.transform(labs)
        
        return labs.astype(np.float32)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        image = self.transform(image)
        
        lab = torch.tensor(self.lab_values[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'lab': lab,
            'label': label
        }


def load_dataset_from_folder(data_dir):
    """
    Load image paths and labels from folder structure.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding labels
    """
    image_paths = []
    labels = []
    
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_id)
    
    return image_paths, labels


def create_data_loaders(data_dir=None, batch_size=BATCH_SIZE, use_synthetic=False,
                        n_synthetic_samples=200):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to data folder (if using real data)
        batch_size: Batch size for data loaders
        use_synthetic: Whether to use synthetic data
        n_synthetic_samples: Samples per class for synthetic data
        
    Returns:
        train_loader, val_loader, test_loader, lab_normalizer
    """
    # Initialize lab normalizer
    lab_gen = LabValueGenerator()
    stats = lab_gen.get_normalization_stats()
    lab_normalizer = LabNormalizer(mean=stats['mean'], std=stats['std'])
    
    if use_synthetic:
        # Create synthetic datasets
        full_dataset = SyntheticCTDataset(
            n_samples_per_class=n_synthetic_samples,
            lab_normalizer=lab_normalizer,
            augment=False
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(TRAIN_RATIO * total_size)
        val_size = int(VAL_RATIO * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(SEED)
        )
    else:
        # Load real data
        if data_dir is None:
            raise ValueError("data_dir must be provided when not using synthetic data")
        
        image_paths, labels = load_dataset_from_folder(data_dir)
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Shuffle and split
        indices = np.arange(len(image_paths))
        np.random.seed(SEED)
        np.random.shuffle(indices)
        
        train_end = int(TRAIN_RATIO * len(indices))
        val_end = int((TRAIN_RATIO + VAL_RATIO) * len(indices))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create datasets
        train_dataset = CTLabDataset(
            [image_paths[i] for i in train_indices],
            [labels[i] for i in train_indices],
            lab_normalizer=lab_normalizer,
            augment=True
        )
        
        val_dataset = CTLabDataset(
            [image_paths[i] for i in val_indices],
            [labels[i] for i in val_indices],
            lab_normalizer=lab_normalizer,
            augment=False
        )
        
        test_dataset = CTLabDataset(
            [image_paths[i] for i in test_indices],
            [labels[i] for i in test_indices],
            lab_normalizer=lab_normalizer,
            augment=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, lab_normalizer


# =============================================================================
# UNIFIED MULTIMODAL DATASET - Combines COVID, Kidney, Lungs datasets
# =============================================================================
# Class mapping:
#   0 = Normal (healthy) - non-COVID + no_nodules lungs
#   1 = Tumor - lung nodules
#   2 = Infection - COVID positive
#   3 = Inflammatory - kidney cysts

class UnifiedMultimodalDataset(Dataset):
    """
    Unified dataset combining:
    - COVID CT scans (Infection vs Normal)
    - Kidney CT scans (Inflammatory/cysts)
    - Lung CT scans (Tumor/nodules vs Normal)
    
    With real NHANES blood report data or synthetic lab values.
    """
    
    def __init__(self, dataset_root, lab_normalizer=None, transform=None, 
                 augment=False, include_covid=True, include_kidney=True, 
                 include_lungs=False, max_samples_per_class=None,
                 use_real_labs=True):
        """
        Args:
            dataset_root: Path to dataset folder containing ctscan_covid, ctscans_kidney, ctscans_lungs
            lab_normalizer: LabNormalizer instance
            transform: Image transforms
            augment: Whether to apply augmentation
            include_covid: Include COVID dataset
            include_kidney: Include Kidney dataset  
            include_lungs: Include Lungs DICOM dataset (slower to load)
            max_samples_per_class: Limit samples per class for balanced training
        """
        self.dataset_root = dataset_root
        self.lab_normalizer = lab_normalizer
        self.augment = augment
        self.max_samples_per_class = max_samples_per_class
        self.use_real_labs = use_real_labs
        
        self.image_paths = []
        self.labels = []
        self.sources = []  # Track which dataset each sample came from
        
        # Load datasets
        if include_covid:
            self._load_covid_data()
        if include_kidney:
            self._load_kidney_data()
        if include_lungs and PYDICOM_AVAILABLE:
            self._load_lungs_data()
        
        # Balance classes if requested
        if max_samples_per_class:
            self._balance_classes()
        
        # Setup lab values - either real NHANES data or synthetic
        if use_real_labs:
            self.nhanes_loader = NHANESLabLoader(dataset_root, seed=SEED)
            self.lab_values = self._generate_real_lab_values()
            print("Using REAL NHANES blood report data")
        else:
            self.lab_generator = LabValueGenerator(seed=SEED)
            self.lab_values = self._generate_synthetic_lab_values()
            print("Using synthetic lab values")
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
        
        print(f"Unified Dataset loaded: {len(self)} samples")
        self._print_class_distribution()
    
    def _load_covid_data(self):
        """Load COVID CT scans: COVID->Infection(2), non-COVID->Normal(0)"""
        covid_dir = os.path.join(self.dataset_root, 'ctscan_covid')
        
        # COVID positive -> Infection (class 2)
        covid_path = os.path.join(covid_dir, 'COVID')
        if os.path.exists(covid_path):
            for img_name in os.listdir(covid_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(covid_path, img_name))
                    self.labels.append(2)  # Infection
                    self.sources.append('covid')
        
        # Non-COVID -> Normal (class 0)
        non_covid_path = os.path.join(covid_dir, 'non-COVID')
        if os.path.exists(non_covid_path):
            for img_name in os.listdir(non_covid_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(non_covid_path, img_name))
                    self.labels.append(0)  # Normal
                    self.sources.append('covid')
    
    def _load_kidney_data(self):
        """Load Kidney CT scans: all kidney scans with detected cysts -> Inflammatory(3)"""
        kidney_dir = os.path.join(self.dataset_root, 'ctscans_kidney')
        
        for split in ['train', 'valid', 'test']:
            images_dir = os.path.join(kidney_dir, split, 'images')
            labels_dir = os.path.join(kidney_dir, split, 'labels')
            
            if not os.path.exists(images_dir):
                continue
                
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(images_dir, img_name)
                    label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
                    
                    # Check if has bounding box (cyst detected)
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                            if content:  # Has annotations -> Inflammatory
                                self.image_paths.append(img_path)
                                self.labels.append(3)  # Inflammatory (cyst)
                                self.sources.append('kidney')
    
    def _load_lungs_data(self):
        """Load Lung DICOM scans: nodules->Tumor(1), no_nodules->Normal(0)"""
        lungs_dir = os.path.join(self.dataset_root, 'ctscans_lungs')
        
        # This is slower - loads DICOM files
        # For now, we'll just scan the folder structure
        nodules_dir = os.path.join(lungs_dir, 'nodules', 'LIDC-IDRI')
        no_nodules_dir = os.path.join(lungs_dir, 'no_nodules', 'manifest-1769675599632', 'LIDC-IDRI')
        
        # Find DICOM files in nodules folder
        if os.path.exists(nodules_dir):
            for patient in os.listdir(nodules_dir)[:50]:  # Limit for speed
                patient_path = os.path.join(nodules_dir, patient)
                if os.path.isdir(patient_path):
                    dcm_files = self._find_dicom_files(patient_path)
                    for dcm_path in dcm_files[:5]:  # Limit per patient
                        self.image_paths.append(dcm_path)
                        self.labels.append(1)  # Tumor
                        self.sources.append('lungs_dicom')
        
        # Find DICOM files in no_nodules folder
        if os.path.exists(no_nodules_dir):
            for patient in os.listdir(no_nodules_dir)[:50]:
                patient_path = os.path.join(no_nodules_dir, patient)
                if os.path.isdir(patient_path):
                    dcm_files = self._find_dicom_files(patient_path)
                    for dcm_path in dcm_files[:5]:
                        self.image_paths.append(dcm_path)
                        self.labels.append(0)  # Normal
                        self.sources.append('lungs_dicom')
    
    def _find_dicom_files(self, root_path, max_files=10):
        """Recursively find DICOM files"""
        dcm_files = []
        for root, dirs, files in os.walk(root_path):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_files.append(os.path.join(root, f))
                    if len(dcm_files) >= max_files:
                        return dcm_files
        return dcm_files
    
    def _balance_classes(self):
        """Balance dataset by limiting samples per class"""
        from collections import defaultdict
        class_indices = defaultdict(list)
        
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        # Select balanced indices
        balanced_indices = []
        for label, indices in class_indices.items():
            np.random.seed(SEED)
            np.random.shuffle(indices)
            balanced_indices.extend(indices[:self.max_samples_per_class])
        
        # Reorder
        self.image_paths = [self.image_paths[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]
        self.sources = [self.sources[i] for i in balanced_indices]
    
    def _generate_real_lab_values(self):
        """Assign real NHANES lab values based on disease class"""
        labs = []
        fallback_count = 0
        
        # Initialize synthetic fallback generator
        fallback_generator = LabValueGenerator(seed=SEED)
        
        for label in self.labels:
            real_lab = self.nhanes_loader.get_lab_for_class(label)
            
            if real_lab is not None:
                labs.append(real_lab)
            else:
                # Fallback to synthetic if category depleted
                fallback_count += 1
                lab = fallback_generator.generate_for_class_id(label, n_samples=1)[0]
                labs.append(lab)
        
        if fallback_count > 0:
            print(f"  Used {fallback_count} synthetic fallback samples (real data exhausted)")
        
        labs = np.array(labs)
        
        if self.lab_normalizer is not None:
            labs = self.lab_normalizer.transform(labs)
        
        return labs.astype(np.float32)
    
    def _generate_synthetic_lab_values(self):
        """Generate synthetic lab values based on disease class"""
        labs = []
        for label in self.labels:
            lab = self.lab_generator.generate_for_class_id(label, n_samples=1)[0]
            labs.append(lab)
        labs = np.array(labs)
        
        if self.lab_normalizer is not None:
            labs = self.lab_normalizer.transform(labs)
        
        return labs.astype(np.float32)
    
    def _get_default_transform(self):
        """Default image transforms"""
        if self.augment:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[MEAN], std=[STD])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[MEAN], std=[STD])
            ])
    
    def _print_class_distribution(self):
        """Print class distribution"""
        from collections import Counter
        counts = Counter(self.labels)
        print("Class distribution:")
        for label, count in sorted(counts.items()):
            print(f"  {CLASS_NAMES[label]}: {count}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        source = self.sources[idx]
        
        # Load image based on source type
        if source == 'lungs_dicom' and PYDICOM_AVAILABLE:
            # Load DICOM
            dcm = pydicom.dcmread(img_path)
            img_array = dcm.pixel_array.astype('float32')
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array).convert('L')
        else:
            # Load regular image
            image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        lab = torch.tensor(self.lab_values[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'lab': lab,
            'label': label,
            'path': img_path
        }


def create_unified_loaders(dataset_root, batch_size=BATCH_SIZE, include_lungs=False, 
                           max_samples_per_class=None, use_real_labs=True):
    """
    Create train/val/test loaders for unified multimodal dataset.
    
    Args:
        dataset_root: Path to dataset folder
        batch_size: Batch size
        include_lungs: Whether to include DICOM lung scans
        max_samples_per_class: Limit samples per class
        use_real_labs: Use real NHANES blood report data (default: True)
    
    Returns:
        train_loader, val_loader, test_loader, lab_normalizer
    """
    # Initialize lab normalizer based on data source
    if use_real_labs:
        # Use NHANES data statistics for normalization
        nhanes_loader = NHANESLabLoader(dataset_root)
        stats = nhanes_loader.get_normalization_stats()
        if stats is not None:
            lab_normalizer = LabNormalizer(mean=stats['mean'], std=stats['std'])
            print(f"Lab normalization from NHANES: mean={stats['mean']}, std={stats['std']}")
        else:
            # Fallback to synthetic stats
            lab_gen = LabValueGenerator()
            stats = lab_gen.get_normalization_stats()
            lab_normalizer = LabNormalizer(mean=stats['mean'], std=stats['std'])
    else:
        lab_gen = LabValueGenerator()
        stats = lab_gen.get_normalization_stats()
        lab_normalizer = LabNormalizer(mean=stats['mean'], std=stats['std'])
    
    # Create full dataset
    full_dataset = UnifiedMultimodalDataset(
        dataset_root=dataset_root,
        lab_normalizer=lab_normalizer,
        augment=False,
        include_covid=True,
        include_kidney=True,
        include_lungs=include_lungs,
        max_samples_per_class=max_samples_per_class,
        use_real_labs=use_real_labs
    )
    
    # Split dataset
    total_size = len(full_dataset)
    indices = np.arange(total_size)
    np.random.seed(SEED)
    np.random.shuffle(indices)
    
    train_end = int(TRAIN_RATIO * total_size)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * total_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Unified Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, lab_normalizer


if __name__ == "__main__":
    # Test synthetic data loading
    print("Testing Synthetic Data Loading...")
    train_loader, val_loader, test_loader, normalizer = create_data_loaders(
        use_synthetic=True, n_synthetic_samples=100
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Labs shape: {batch['lab'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"  Labels: {batch['label'][:5]}")
