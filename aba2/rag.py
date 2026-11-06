
from pathlib import Path
 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KB_SOURCE = "Project knowledge base (educational summary - verify against MedlinePlus Lab Tests)"
 
 
def _doc(doc_id, title, text):
    return {"id": doc_id, "title": title, "text": text, "source": KB_SOURCE}
 
 
KNOWLEDGE_BASE = [
    # ---------------- General guidance ----------------
    _doc("gen_reference_ranges", "Understanding reference ranges",
         "A reference range is the range of values found in most healthy people, usually the middle 95%. "
         "This means about 1 in 20 healthy people will have a result slightly outside the range on any single test. "
         "Ranges differ between laboratories, testing methods, age, sex and pregnancy, so the range printed on "
         "your own report is the one to use."),
    _doc("gen_not_diagnosis", "Why one abnormal value is not a diagnosis",
         "A single value outside the reference range is not a diagnosis. Doctors interpret results together with "
         "symptoms, medical history, medicines, examination and trends over time, and mildly abnormal values are "
         "often simply repeated. Results can be affected by recent food, exercise, hydration, illness, medicines, "
         "time of day and how the sample was handled."),
    _doc("gen_urgent", "When to seek prompt medical care",
         "Some situations need prompt medical attention whatever the lab values show: chest pain, difficulty "
         "breathing, fainting, sudden confusion, severe bleeding, very high fever or signs of a severe allergic "
         "reaction. Laboratories usually phone the doctor directly about dangerously abnormal 'critical values'."),
    _doc("gen_cbc", "Complete blood count (CBC) overview",
         "A complete blood count looks at the three main types of blood cells: red blood cells that carry oxygen "
         "(hemoglobin, RBC count, hematocrit, MCV), white blood cells that fight infection (WBC and differential "
         "count), and platelets that help blood clot. It is one of the most common screening tests."),
    _doc("gen_lipid", "Lipid profile overview",
         "A lipid profile measures fats in the blood: total cholesterol, LDL, HDL, triglycerides and often VLDL. "
         "It helps estimate risk of heart and blood vessel disease together with age, blood pressure, smoking, "
         "diabetes and family history. Some labs require fasting before the test, which mainly affects triglycerides."),
    _doc("gen_lft", "Liver function tests (LFT) overview",
         "Liver function tests include enzymes such as ALT (SGPT), AST (SGOT) and ALP, plus bilirubin, albumin and "
         "total protein. Together they show whether liver cells are irritated, whether bile flow is affected and how "
         "well the liver makes proteins. The pattern across tests matters more than any single value."),
    _doc("gen_kft", "Kidney function tests (KFT/RFT) overview",
         "Kidney function tests usually include creatinine, urea or BUN, uric acid and electrolytes such as sodium "
         "and potassium. Creatinine is often used with age and sex to estimate the glomerular filtration rate (eGFR), "
         "which shows how well the kidneys filter blood."),
    _doc("gen_thyroid", "Thyroid profile overview",
         "A thyroid profile usually includes TSH, T3 and T4. TSH from the pituitary gland controls the thyroid, so "
         "TSH and thyroid hormones normally move in opposite directions. Results can be influenced by illness, "
         "pregnancy and some medicines."),
 
    # ---------------- Blood count ----------------
    _doc("hemoglobin", "Hemoglobin (Hb)",
         "Hemoglobin is the iron-containing protein in red blood cells that carries oxygen around the body. A value "
         "below the reference range is called anemia; common general reasons include iron deficiency, vitamin B12 or "
         "folate deficiency, blood loss, heavy periods or long-term illness. Low hemoglobin can cause tiredness, "
         "weakness, breathlessness or pale skin, though mild lows may cause no symptoms. A high value can be seen with "
         "dehydration, smoking, living at high altitude, or some lung, heart or bone marrow conditions."),
    _doc("rbc", "Red blood cell (RBC) count",
         "The RBC count is the number of red blood cells in a volume of blood. Low counts usually go together with low "
         "hemoglobin, for reasons such as nutritional deficiency, blood loss or chronic disease. High counts can occur "
         "with dehydration, smoking, high altitude or conditions where the body makes too many red cells. It is read "
         "together with hemoglobin, hematocrit and MCV."),
    _doc("wbc", "White blood cell count (WBC / Total leucocyte count, TLC)",
         "White blood cells are part of the immune system. A high count (leukocytosis) is most often a response to "
         "infection or inflammation, but can also follow physical stress, steroid medicines or smoking, and less "
         "commonly blood disorders. A low count (leukopenia) can follow viral infections, some medicines, nutritional "
         "deficiencies or bone marrow problems. The differential count shows which type of white cell has changed."),
    _doc("neutrophils", "Neutrophils",
         "Neutrophils are the most common white blood cells and mainly fight bacterial infections. They are often "
         "raised in bacterial infection, inflammation, physical stress or with steroid use. They may be low after some "
         "viral infections, with certain medicines or bone marrow problems. Reports may show a percentage and/or an "
         "absolute count."),
    _doc("lymphocytes", "Lymphocytes",
         "Lymphocytes are white blood cells (B cells and T cells) that fight viral infections and make antibodies. They "
         "are often raised during or after viral infections. They can be low during acute stress, with steroid use, or "
         "in some infections and immune conditions."),
    _doc("platelets", "Platelet count",
         "Platelets are small cell fragments that help blood clot. A low count (thrombocytopenia) can occur with viral "
         "infections such as dengue, some medicines, immune conditions, liver disease or bone marrow problems; very low "
         "counts raise the risk of bleeding. A high count is often a reaction to infection, inflammation, iron "
         "deficiency or blood loss, and less commonly a bone marrow condition. Indian reports often write platelets in "
         "lakhs per cumm (1.5 lakhs = 150,000)."),
    _doc("hematocrit", "Hematocrit (PCV, packed cell volume)",
         "Hematocrit, also called packed cell volume (PCV), is the percentage of blood made up of red blood cells. It "
         "usually moves in the same direction as hemoglobin: low values suggest anemia and high values can be seen with "
         "dehydration or conditions that cause too many red cells."),
    _doc("mcv", "MCV (mean corpuscular volume)",
         "MCV is the average size of red blood cells. A low MCV (small cells) is commonly seen with iron deficiency or "
         "thalassemia trait. A high MCV (large cells) can be seen with vitamin B12 or folate deficiency, alcohol use, "
         "liver disease, thyroid problems or some medicines. Doctors use MCV to work out the likely type of anemia."),
    _doc("esr", "ESR (erythrocyte sedimentation rate)",
         "ESR measures how fast red cells settle in a tube and is a non-specific sign of inflammation. It can be raised "
         "by infections, inflammatory or autoimmune conditions, anemia, pregnancy and ageing. A raised ESR does not "
         "point to one cause and is interpreted with symptoms and other tests."),
    _doc("crp", "CRP (C-reactive protein)",
         "CRP is a protein made by the liver that rises with inflammation or infection. A raised CRP shows there is "
         "inflammation somewhere in the body but not where or why. High-sensitivity CRP (hs-CRP) is a related test "
         "sometimes used in heart-risk assessment."),
 
    # ---------------- Sugar ----------------
    _doc("glucose_fasting", "Fasting blood sugar (FBS)",
         "Fasting blood glucose measures blood sugar after at least 8 hours without food. Values above the range can be "
         "seen in prediabetes or diabetes, but also with stress, illness, some medicines, or if the person was not truly "
         "fasting. Diabetes is not diagnosed from one reading; doctors usually confirm with a repeat test or HbA1c. Low "
         "values (hypoglycemia) can cause shakiness, sweating or confusion."),
    _doc("glucose_pp", "Post-prandial blood sugar (PPBS)",
         "Post-prandial blood sugar is measured about 2 hours after a meal and shows how the body handles sugar after "
         "eating. Values above the range can suggest impaired glucose handling and are interpreted with fasting glucose "
         "and HbA1c."),
    _doc("glucose_random", "Random blood sugar (RBS)",
         "Random blood sugar is measured at any time regardless of meals. Because it depends on what was eaten recently, "
         "a high value is usually followed up with fasting glucose or HbA1c."),
    _doc("hba1c", "HbA1c (glycated hemoglobin)",
         "HbA1c reflects the average blood sugar level over roughly the past 2 to 3 months. Higher values mean higher "
         "average blood sugar and are used to screen for and monitor prediabetes and diabetes. Results can be affected "
         "by anemia, some hemoglobin variants, recent blood loss or transfusion."),
 
    # ---------------- Lipids ----------------
    _doc("total_cholesterol", "Total cholesterol",
         "Total cholesterol is the overall cholesterol in the blood, including LDL and HDL. High values can be linked to "
         "diet, inactivity, excess weight, genetics, an underactive thyroid, diabetes or some medicines. It is one factor "
         "in heart risk, judged together with the rest of the lipid profile."),
    _doc("ldl", "LDL cholesterol",
         "LDL is often called 'bad' cholesterol because high levels add to plaque build-up in arteries. Raised LDL can be "
         "related to diet high in saturated fat, inactivity, excess weight, genetics such as familial "
         "hypercholesterolemia, and some medical conditions. The right target depends on a person's overall heart risk."),
    _doc("hdl", "HDL cholesterol",
         "HDL is often called 'good' cholesterol because it helps carry cholesterol away from the arteries. Low HDL is "
         "linked to higher heart risk and can be associated with inactivity, smoking, excess weight, high triglycerides "
         "and genetics. Unlike most tests, a higher HDL is generally considered favourable."),
    _doc("vldl", "VLDL cholesterol",
         "VLDL carries triglycerides in the blood and is usually calculated from the triglyceride value. It tends to rise "
         "and fall with triglycerides."),
    _doc("triglycerides", "Triglycerides",
         "Triglycerides are fats in the blood that store energy from food. High levels can be related to excess calories, "
         "sugar or alcohol, excess weight, inactivity, uncontrolled diabetes, an underactive thyroid, kidney disease or "
         "some medicines. Not fasting before the test can raise the value. Very high levels increase the risk of "
         "pancreatitis."),
 
    # ---------------- Kidney ----------------
    _doc("creatinine", "Serum creatinine",
         "Creatinine is a waste product of normal muscle activity that the kidneys filter out. A raised value can mean the "
         "kidneys are filtering less well, but it is also affected by dehydration, high muscle mass, heavy meat intake or "
         "some medicines. Low values can occur with low muscle mass. It is used with age and sex to estimate eGFR."),
    _doc("urea", "Blood urea / BUN",
         "Urea (or blood urea nitrogen, BUN) is a waste product of protein breakdown removed by the kidneys. High levels "
         "can occur with reduced kidney function, dehydration, high protein intake, bleeding in the gut or some "
         "medicines. Low levels can be seen with low protein intake or liver disease. It is read with creatinine."),
    _doc("uric_acid", "Uric acid",
         "Uric acid is produced when the body breaks down purines from food and its own cells. High levels can be linked "
         "to red meat, seafood, alcohol, sugary drinks, excess weight, kidney function, some medicines and genetics. It "
         "can lead to gout or kidney stones in some people, although many people with high uric acid have no symptoms."),
 
    # ---------------- Liver ----------------
    _doc("alt", "ALT / SGPT",
         "ALT (alanine aminotransferase, also called SGPT) is an enzyme found mainly in the liver. A raised ALT suggests "
         "liver cells may be irritated or injured, for reasons such as fatty liver, alcohol, viral hepatitis, some "
         "medicines or supplements, or recent heavy exercise. Mild rises are common and are often rechecked."),
    _doc("ast", "AST / SGOT",
         "AST (aspartate aminotransferase, also called SGOT) is an enzyme found in the liver, heart and muscles. A raised "
         "AST can reflect liver irritation but also muscle injury or strenuous exercise. Doctors often compare AST with "
         "ALT to understand the likely source."),
    _doc("alp", "Alkaline phosphatase (ALP)",
         "ALP is an enzyme found mainly in the liver and bones. It may be raised when bile flow from the liver is "
         "affected or in some bone conditions. It is normally higher in growing children, teenagers and in pregnancy."),
    _doc("bilirubin", "Bilirubin",
         "Bilirubin is a yellow pigment made when old red blood cells break down, and it is processed by the liver. High "
         "bilirubin can cause jaundice and may result from liver conditions, blocked bile flow, faster breakdown of red "
         "cells, or Gilbert's syndrome, a common harmless condition. Direct and indirect bilirubin help locate the cause."),
    _doc("albumin", "Albumin",
         "Albumin is the main protein made by the liver; it keeps fluid inside blood vessels and carries substances. Low "
         "albumin can occur with liver disease, kidney conditions that leak protein, poor nutrition or inflammation. High "
         "albumin is uncommon and usually reflects dehydration."),
 
    # ---------------- Thyroid ----------------
    _doc("tsh", "TSH (thyroid stimulating hormone)",
         "TSH is made by the pituitary gland and tells the thyroid how much hormone to make. A high TSH often means the "
         "thyroid is underactive (hypothyroidism); a low TSH often means it is overactive (hyperthyroidism). TSH can also "
         "be affected by illness, pregnancy, some medicines and time of day, so mild changes are often rechecked with T4."),
    _doc("t3", "T3 (triiodothyronine)",
         "T3 is an active thyroid hormone. It is interpreted with TSH and T4; it may be high in an overactive thyroid and "
         "can be low during illness or with an underactive thyroid. Free T3 measures the unbound, active form."),
    _doc("t4", "T4 (thyroxine)",
         "T4 is the main hormone made by the thyroid gland. Low T4 with a high TSH suggests an underactive thyroid; high "
         "T4 with a low TSH suggests an overactive thyroid. Free T4 measures the unbound, active form."),
 
    # ---------------- Vitamins, iron, minerals ----------------
    _doc("vitamin_d", "Vitamin D (25-hydroxy vitamin D)",
         "Vitamin D helps the body absorb calcium and keeps bones and muscles healthy. Low levels are very common and can "
         "be related to limited sunlight, darker skin, diet, excess weight or digestive conditions; deficiency can "
         "contribute to bone pain, muscle weakness and tiredness. Very high levels are uncommon and usually come from "
         "taking too many supplements."),
    _doc("vitamin_b12", "Vitamin B12",
         "Vitamin B12 is needed to make red blood cells and keep nerves healthy. Low levels can be due to vegetarian or "
         "vegan diets, poor absorption from the gut, long-term use of some medicines such as metformin or acid-reducing "
         "drugs, and older age. Deficiency can cause anemia with large red cells, tiredness, tingling or numbness."),
    _doc("ferritin", "Ferritin",
         "Ferritin is the protein that stores iron, and its blood level reflects the body's iron stores. Low ferritin is "
         "one of the most reliable signs of iron deficiency. Ferritin also rises with inflammation, infection, liver "
         "disease or iron overload, so a normal value does not always rule out low iron."),
    _doc("iron", "Serum iron",
         "Serum iron measures iron circulating in the blood. It changes through the day and after meals, so it is read "
         "together with ferritin, TIBC and transferrin saturation to judge iron status."),
    _doc("sodium", "Sodium",
         "Sodium is an electrolyte that controls fluid balance and helps nerves and muscles work. Low sodium can occur with "
         "vomiting, diarrhea, some medicines such as diuretics, drinking a lot of water, or heart, liver or kidney "
         "conditions. High sodium is usually due to dehydration. Large changes can cause confusion or weakness and need "
         "medical attention."),
    _doc("potassium", "Potassium",
         "Potassium is an electrolyte essential for heart rhythm, nerves and muscles. Low potassium can result from "
         "vomiting, diarrhea or diuretic medicines. High potassium can occur with kidney problems or some medicines, and "
         "can be falsely high if the sample was damaged (hemolysed). Clearly abnormal potassium can affect heart rhythm "
         "and needs prompt medical attention."),
    _doc("calcium", "Calcium",
         "Calcium is important for bones, muscles, nerves and the heart. Low calcium can relate to low vitamin D, low "
         "albumin, kidney disease or parathyroid problems. High calcium is most often due to an overactive parathyroid "
         "gland. Doctors often correct total calcium for albumin."),
]
