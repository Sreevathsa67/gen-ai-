import io
import re
 
import pdfplumber
 
# ---------------------------------------------------------------------------
# 1. Test-name normalisation (canonical id -> aliases seen on Indian lab reports)
#    Canonical ids match the document ids in rag.py's knowledge base.
# ---------------------------------------------------------------------------
ALIASES = {
    "hemoglobin": ["hemoglobin", "haemoglobin", "hb", "hgb"],
    "rbc": ["rbc", "rbc count", "total rbc count", "red blood cell count", "red blood cells",
            "erythrocyte count", "red cell count"],
    "wbc": ["wbc", "wbc count", "total wbc count", "white blood cell count", "total leucocyte count",
            "total leukocyte count", "tlc", "leucocyte count", "leukocyte count", "total count"],
    "neutrophils": ["neutrophils", "neutrophil", "polymorphs", "neutrophil count"],
    "lymphocytes": ["lymphocytes", "lymphocyte"],
    "platelets": ["platelet count", "platelets", "platelet", "plt"],
    "hematocrit": ["hematocrit", "haematocrit", "pcv", "packed cell volume", "hct"],
    "mcv": ["mcv", "mean corpuscular volume", "mean cell volume"],
    "esr": ["esr", "erythrocyte sedimentation rate"],
    "crp": ["crp", "c reactive protein", "hs crp"],
    "glucose_fasting": ["fasting blood sugar", "fbs", "fasting blood glucose", "glucose fasting",
                        "fasting glucose", "fasting plasma glucose", "fpg", "blood sugar fasting",
                        "plasma glucose fasting"],
    "glucose_pp": ["post prandial blood sugar", "postprandial blood sugar", "ppbs", "pp blood sugar",
                   "blood sugar pp", "glucose pp", "post prandial glucose", "plasma glucose pp"],
    "glucose_random": ["random blood sugar", "rbs", "random glucose", "blood sugar random",
                       "glucose random"],
    "hba1c": ["hba1c", "glycated hemoglobin", "glycosylated hemoglobin", "glycated haemoglobin",
              "glycosylated haemoglobin", "a1c"],
    "total_cholesterol": ["total cholesterol", "cholesterol total", "serum cholesterol", "cholesterol"],
    "ldl": ["ldl", "ldl cholesterol", "ldl c", "low density lipoprotein"],
    "hdl": ["hdl", "hdl cholesterol", "hdl c", "high density lipoprotein"],
    "vldl": ["vldl", "vldl cholesterol"],
    "triglycerides": ["triglycerides", "triglyceride", "serum triglycerides", "tg"],
    "creatinine": ["creatinine", "serum creatinine", "s creatinine"],
    "urea": ["urea", "blood urea", "serum urea", "bun", "blood urea nitrogen"],
    "uric_acid": ["uric acid", "serum uric acid"],
    "alt": ["alt", "sgpt", "alanine aminotransferase"],
    "ast": ["ast", "sgot", "aspartate aminotransferase"],
    "alp": ["alp", "alkaline phosphatase"],
    "bilirubin": ["bilirubin", "total bilirubin", "bilirubin total", "serum bilirubin",
                  "direct bilirubin", "bilirubin direct", "indirect bilirubin"],
    "albumin": ["albumin", "serum albumin"],
    "tsh": ["tsh", "thyroid stimulating hormone", "ultrasensitive tsh", "us tsh"],
    "t3": ["t3", "total t3", "free t3", "ft3", "triiodothyronine"],
    "t4": ["t4", "total t4", "free t4", "ft4", "thyroxine"],
    "vitamin_d": ["vitamin d", "vit d", "vitamin d3", "25 oh vitamin d", "25 hydroxy vitamin d",
                  "vitamin d 25 oh", "vitamin d total"],
    "vitamin_b12": ["vitamin b12", "vit b12", "b12", "cobalamin", "cyanocobalamin"],
    "ferritin": ["ferritin", "serum ferritin"],
    "iron": ["iron", "serum iron"],
    "sodium": ["sodium", "serum sodium"],
    "potassium": ["potassium", "serum potassium"],
    "calcium": ["calcium", "serum calcium", "total calcium"],
}
