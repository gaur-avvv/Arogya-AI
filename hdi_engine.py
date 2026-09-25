"""
Herb-Drug Interaction (HDI) & Clinical Contraindication Safety Engine
=====================================================================

Provides clinical pharmacovigilance screening between modern allopathic pharmaceuticals
and classical Ayurvedic herbs, formulations, and patient allergies.

Classifies interactions into clinical severity tiers:
- HIGH: Potentially severe adverse events, direct contraindication, or marked toxicity risk.
- MODERATE: Clinically significant interaction requiring staggered dosing or biochemical monitoring.
- CAUTION: Mild additive physiological effect or mild pharmacodynamic synergy.
- SAFE: No documented adverse interactions.
"""

from typing import List, Dict, Any
import re

# Comprehensive evidence-based interaction matrix
HDI_KNOWLEDGE_BASE = [
    {
        "drug_category": "Anticoagulants & Antiplatelets",
        "drugs": ["warfarin", "aspirin", "clopidogrel", "heparin", "apixaban", "rivaroxaban", "dabigatran", "ecosprin"],
        "herbs": ["guggulu", "shallaki", "ginger", "sunthi", "lasuna", "garlic", "turmeric", "haridra", "curcumin", "ginkgo"],
        "severity": "HIGH",
        "mechanism": "Synergistic inhibition of platelet aggregation and thromboxane synthesis, increasing risk of spontaneous hemorrhage and elevating prothrombin time/INR.",
        "clinical_advice": "Concurrent use contraindicated without strict INR/coagulation monitoring. Advise practitioner to substitute or adjust anticoagulant dosage."
    },
    {
        "drug_category": "Oral Hypoglycemics & Insulin",
        "drugs": ["metformin", "glimepiride", "glipizide", "insulin", "dapagliflozin", "empagliflozin", "sitagliptin", "pioglitazone"],
        "herbs": ["karela", "bitter gourd", "meshashringi", "gymnema", "vijaysar", "gurmar", "methi", "fenugreek", "guduchi"],
        "severity": "MODERATE",
        "mechanism": "Additive hypoglycemic pharmacodynamics accelerating glucose uptake and enhancing pancreatic beta-cell insulin secretion, risking acute hypoglycemia.",
        "clinical_advice": "Closely monitor fasting and postprandial capillary blood glucose. Educate patient on early hypoglycemia symptoms (tremors, sweating, dizziness)."
    },
    {
        "drug_category": "Antihypertensives & ACE Inhibitors",
        "drugs": ["amlodipine", "losartan", "telmisartan", "enalapril", "lisinopril", "atenolol", "metoprolol", "ramipril"],
        "herbs": ["yashtimadhu", "licorice", "mulethi", "glycyrrhiza"],
        "severity": "HIGH",
        "mechanism": "Glycyrrhizin inhibits 11-beta-hydroxysteroid dehydrogenase type 2, promoting pseudoaldosteronism, potassium wasting, sodium retention, and secondary blood pressure elevation.",
        "clinical_advice": "Do not co-administer licorice preparations with antihypertensives. Use DGL (deglycyrrhizinated licorice) or substitute with Shatavari."
    },
    {
        "drug_category": "Antihypertensives & Beta-Blockers (Hypotensive synergy)",
        "drugs": ["amlodipine", "losartan", "telmisartan", "atenolol", "metoprolol", "ramipril"],
        "herbs": ["sarpagandha", "rauwolfia", "arjuna", "jatamansi"],
        "severity": "MODERATE",
        "mechanism": "Additive peripheral vasodilation and sympatholytic action causing postural hypotension, bradycardia, or syncope.",
        "clinical_advice": "Monitor blood pressure when introducing cardiac or calming herbs. Dose with at least a 2-hour interval."
    },
    {
        "drug_category": "CNS Depressants, Benzodiazepines & Sedatives",
        "drugs": ["alprazolam", "clonazepam", "diazepam", "lorazepam", "zolpidem", "phenobarbital", "eszopiclone"],
        "herbs": ["ashwagandha", "jatamansi", "tagara", "valerian", "brahmi", "shankhpushpi"],
        "severity": "MODERATE",
        "mechanism": "GABAergic and central nervous system depressant potentiation, leading to pronounced somnolence, cognitive slowing, and psychomotor impairment.",
        "clinical_advice": "Advise avoiding high sedative herbal doses when taking prescription benzodiazepines. Caution against driving or operating machinery."
    },
    {
        "drug_category": "Antidepressants & SSRIs / SNRIs",
        "drugs": ["sertraline", "fluoxetine", "escitalopram", "paroxetine", "duloxetine", "venlafaxine"],
        "herbs": ["jatamansi", "shankhpushpi", "st. john's wort"],
        "severity": "MODERATE",
        "mechanism": "Potential pharmacodynamic elevation of synaptic serotonin and monoamine concentrations, raising theoretical risk of serotonin syndrome.",
        "clinical_advice": "Maintain conservative dosing and watch for autonomic instability, agitation, or hyperreflexia."
    },
    {
        "drug_category": "Thyroid Hormone Replacements",
        "drugs": ["levothyroxine", "eltroxin", "synthroid"],
        "herbs": ["ashwagandha", "guggulu", "kanchnar"],
        "severity": "MODERATE",
        "mechanism": "Ashwagandha stimulates endogenous T3/T4 secretion and 5'-monodeiodinase activity, potentially requiring a decrease in exogenous levothyroxine dosage.",
        "clinical_advice": "Schedule TSH/FT4 panel 4 to 6 weeks after initiating therapy to evaluate need for pharmaceutical dose titration."
    },
    {
        "drug_category": "Immunosuppressants & Corticosteroids",
        "drugs": ["prednisone", "dexamethasone", "cyclosporine", "tacrolimus", "methotrexate", "azathioprine"],
        "herbs": ["guduchi", "tinospora", "ashwagandha", "tulasi", "amla"],
        "severity": "HIGH",
        "mechanism": "Herbal immune stimulants activate macrophage phagocytosis and cytokine cascades, directly counteracting pharmacologic immunosuppression in organ transplants or severe autoimmune flares.",
        "clinical_advice": "Avoid potent immunomodulatory herbs (Rasayanas) in organ transplant recipients or active high-dose immunosuppressive therapy."
    },
    {
        "drug_category": "Diuretics & Potassium-depleting agents",
        "drugs": ["furosemide", "hydrochlorothiazide", "spironolactone", "lasix"],
        "herbs": ["punarnava", "gokshura", "yashtimadhu"],
        "severity": "MODERATE",
        "mechanism": "Additive natriuresis and diuresis; Yashtimadhu markedly enhances renal potassium loss, worsening hypokalemia risk.",
        "clinical_advice": "Check baseline serum electrolytes. Advise adequate fluid intake and separate diuretic administration by 3 hours."
    },
    {
        "drug_category": "Cardiac Glycosides",
        "drugs": ["digoxin", "lanoxin"],
        "herbs": ["yashtimadhu", "licorice", "hawthorn", "arjuna"],
        "severity": "HIGH",
        "mechanism": "Licorice-induced hypokalemia markedly potentiates digitalis cardiotoxicity and fatal arrhythmias.",
        "clinical_advice": "Contraindicated. Hypokalemia drastically narrows Digoxin therapeutic index. Use potassium-sparing alternatives."
    }
]

ALLERGY_CONTRAINDICATIONS = [
    {
        "allergen_keywords": ["aspirin", "salicylate", "nsaid"],
        "herbs": ["shallaki", "boswellia", "willow bark", "guggulu"],
        "severity": "HIGH",
        "message": "Potential cross-reactivity with natural botanical salicylates/anti-inflammatories."
    },
    {
        "allergen_keywords": ["ragweed", "asteraceae", "chamomile"],
        "herbs": ["bhringraj", "chamomile", "artemisia", "echinacea"],
        "severity": "MODERATE",
        "message": "Botanical family cross-reactivity for Asteraceae/Compositae pollen allergy."
    }
]

def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

def analyze_herb_drug_interactions(
    current_medications: str,
    recommended_herbs: List[str] or str,
    allergies: str = "None"
) -> Dict[str, Any]:
    """
    Performs algorithmic screening between current patient medications/allergies
    and recommended Ayurvedic botanicals.
    """
    alerts = []
    meds_norm = _normalize(current_medications)
    allergies_norm = _normalize(allergies)

    if isinstance(recommended_herbs, str):
        herbs_list = [h.strip() for h in recommended_herbs.replace('/', ',').replace(';', ',').split(',') if h.strip()]
    else:
        herbs_list = [str(h).strip() for h in recommended_herbs if str(h).strip()]

    has_meds = bool(meds_norm and meds_norm not in ["none", "nil", "na", "no", "nothing"])
    has_allergies = bool(allergies_norm and allergies_norm not in ["none", "nil", "na", "no", "nothing"])

    if not has_meds and not has_allergies:
        return {
            "has_alerts": False,
            "max_severity": "SAFE",
            "total_alerts": 0,
            "alerts": [],
            "summary": "No pharmaceutical medications or botanical allergies reported. Standard Ayurvedic administration applies."
        }

    # 1. Screen Herb-Drug Interactions
    if has_meds:
        for rule in HDI_KNOWLEDGE_BASE:
            matched_drug = None
            for drug in rule["drugs"]:
                # Check whole word match or substring
                pattern = r'\b' + re.escape(drug) + r'\b'
                if re.search(pattern, meds_norm) or drug in meds_norm:
                    matched_drug = drug
                    break

            if matched_drug:
                for herb in herbs_list:
                    herb_norm = _normalize(herb)
                    for target_herb in rule["herbs"]:
                        if target_herb in herb_norm or herb_norm in target_herb:
                            alerts.append({
                                "type": "HERB_DRUG_INTERACTION",
                                "severity": rule["severity"],
                                "drug_matched": matched_drug.title(),
                                "herb_matched": herb.title(),
                                "drug_category": rule["drug_category"],
                                "mechanism": rule["mechanism"],
                                "clinical_advice": rule["clinical_advice"]
                            })

    # 2. Screen Allergy Contraindications
    if allergies_norm and allergies_norm not in ["none", "nil", "na", "no"]:
        for allergy_rule in ALLERGY_CONTRAINDICATIONS:
            matched_allergen = None
            for allergen in allergy_rule["allergen_keywords"]:
                if allergen in allergies_norm:
                    matched_allergen = allergen
                    break

            if matched_allergen:
                for herb in herbs_list:
                    herb_norm = _normalize(herb)
                    for target_herb in allergy_rule["herbs"]:
                        if target_herb in herb_norm:
                            alerts.append({
                                "type": "ALLERGY_CONTRAINDICATION",
                                "severity": allergy_rule["severity"],
                                "drug_matched": matched_allergen.title(),
                                "herb_matched": herb.title(),
                                "drug_category": "Allergy Cross-Reactivity",
                                "mechanism": allergy_rule["message"],
                                "clinical_advice": "Do not administer this herb; substitute with an alternative botanical."
                            })

    # Deduplicate alerts based on (drug_matched, herb_matched)
    unique_alerts = []
    seen = set()
    for alert in alerts:
        key = (alert["drug_matched"].lower(), alert["herb_matched"].lower())
        if key not in seen:
            seen.add(key)
            unique_alerts.append(alert)

    # Determine highest severity
    severities = [a["severity"] for a in unique_alerts]
    if "HIGH" in severities:
        max_sev = "HIGH"
    elif "MODERATE" in severities:
        max_sev = "MODERATE"
    elif "CAUTION" in severities:
        max_sev = "CAUTION"
    else:
        max_sev = "SAFE"

    if max_sev == "HIGH":
        summary = "CRITICAL: Clinically significant herb-drug contraindication detected. Physician review mandatory before herbal administration."
    elif max_sev == "MODERATE":
        summary = "ADVISORY: Potential pharmacological interaction detected. Stagger intake by 2-3 hours and monitor parameters."
    elif max_sev == "CAUTION":
        summary = "NOTICE: Minor botanical synergy detected. Normal precautions apply."
    else:
        summary = "Clear: No documented adverse interactions detected between current medications and recommended herbs."

    return {
        "has_alerts": len(unique_alerts) > 0,
        "max_severity": max_sev,
        "total_alerts": len(unique_alerts),
        "alerts": unique_alerts,
        "summary": summary
    }
