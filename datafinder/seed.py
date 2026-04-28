"""Synthetic biomedical-imaging datasets for demos and CI.

Mirrors the shape of real public datasets the lab worked with —
OAI, ADNI, UKBB, NLST, MRNet, etc. — without copying their
distribution licenses. Fields are realistic enough for the agent
to make meaningful routing + tool-sequencing decisions.
"""

from __future__ import annotations

from datafinder.schema import Dataset
from datafinder.store import Store

_DATASETS: list[Dataset] = [
    Dataset(
        id="ds_oai",
        title="OAI — Knee MRI Cohort",
        description=(
            "Longitudinal magnetic resonance imaging study of knee "
            "osteoarthritis. Includes T1, T2, and DESS sequences with "
            "cartilage thickness annotations and patient-reported "
            "outcomes."
        ),
        modality="MRI",
        anatomy="knee",
        subjects=4796,
        age_min=45, age_max=79,
        annotations=["cartilage_thickness", "WOMAC", "Kellgren-Lawrence"],
        citation="osteoarthritisinitiative.nih.gov",
        columns=["subject_id", "visit", "side", "kl_grade", "cartilage_mm", "womac"],
    ),
    Dataset(
        id="ds_adni",
        title="ADNI — Alzheimer's Brain MRI",
        description=(
            "Structural and functional brain MRI in older adults with "
            "Alzheimer's disease, mild cognitive impairment, and "
            "controls. PET amyloid, tau, and CSF biomarkers attached."
        ),
        modality="MRI",
        anatomy="brain",
        subjects=2400,
        age_min=55, age_max=90,
        annotations=["hippocampus_volume", "MMSE", "amyloid_pet"],
        citation="adni.loni.usc.edu",
        columns=["subject_id", "visit", "diagnosis", "mmse", "hippocampus_ml"],
    ),
    Dataset(
        id="ds_nlst",
        title="NLST — Lung CT Screening",
        description=(
            "National Lung Screening Trial low-dose computed "
            "tomography of heavy smokers. Nodule annotations, "
            "follow-up cancer outcomes."
        ),
        modality="CT",
        anatomy="lung",
        subjects=53454,
        age_min=55, age_max=74,
        annotations=["nodule_segmentation", "lung_rads", "cancer_outcome"],
        citation="cdas.cancer.gov/nlst",
        columns=["subject_id", "scan_date", "nodule_count", "rads_score"],
    ),
    Dataset(
        id="ds_ukbb",
        title="UK Biobank — Multimodal Cohort",
        description=(
            "UK Biobank imaging substudy. Brain MRI, cardiac MRI, DXA "
            "body composition, OCT retinal scans, and full clinical "
            "phenotyping."
        ),
        modality="MRI",
        anatomy="multi-organ",
        subjects=100000,
        age_min=40, age_max=70,
        annotations=["brain_volume", "cardiac_function", "body_composition", "retinal_oct"],
        citation="ukbiobank.ac.uk",
        columns=["eid", "visit", "modality", "field_id", "value"],
    ),
    Dataset(
        id="ds_mrnet",
        title="MRNet — Knee MRI for Tear Detection",
        description=(
            "Stanford knee MRI dataset annotated for ACL tear, "
            "meniscus tear, and abnormality. Sagittal, coronal, and "
            "axial views."
        ),
        modality="MRI",
        anatomy="knee",
        subjects=1370,
        age_min=12, age_max=80,
        annotations=["acl_tear", "meniscus_tear", "abnormality"],
        citation="stanfordmlgroup.github.io/competitions/mrnet",
        columns=["subject_id", "view", "label_acl", "label_meniscus", "label_abnormal"],
    ),
    Dataset(
        id="ds_chexpert",
        title="CheXpert — Chest Radiograph Labels",
        description=(
            "Stanford chest X-ray dataset labeled for 14 thoracic "
            "pathologies, including atelectasis, cardiomegaly, "
            "edema, pneumothorax, and pleural effusion."
        ),
        modality="X-ray",
        anatomy="chest",
        subjects=65240,
        age_min=18, age_max=95,
        annotations=["atelectasis", "cardiomegaly", "edema", "pneumothorax"],
        citation="stanfordmlgroup.github.io/competitions/chexpert",
        columns=["subject_id", "study", "view", "label_*"],
    ),
    Dataset(
        id="ds_oct_kermany",
        title="OCT — Retinal Disease (Kermany 2018)",
        description=(
            "Optical coherence tomography of the retina labeled for "
            "choroidal neovascularization, diabetic macular edema, "
            "drusen, and normal."
        ),
        modality="OCT",
        anatomy="retina",
        subjects=4686,
        age_min=21, age_max=85,
        annotations=["CNV", "DME", "drusen", "normal"],
        citation="data.mendeley.com/datasets/rscbjbr9sj/3",
        columns=["subject_id", "image_id", "label"],
    ),
    Dataset(
        id="ds_mimic_ecg",
        title="MIMIC-ECG — Critical Care 12-lead ECG",
        description=(
            "Twelve-lead electrocardiogram traces from MIMIC-IV ICU "
            "patients with linked clinical outcomes and rhythm labels."
        ),
        modality="ECG",
        anatomy="heart",
        subjects=18000,
        age_min=18, age_max=99,
        annotations=["rhythm", "stemi", "arrhythmia"],
        citation="physionet.org/content/mimic-iv",
        columns=["subject_id", "stay_id", "lead", "rhythm_label"],
    ),
]


def populate(store: Store) -> int:
    for ds in _DATASETS:
        store.upsert(ds)
    return len(_DATASETS)
