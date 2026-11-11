# Targets used (VERIFY before using results in the paper)

## cardiac_arrest_spain — included
- table: S1 Dataset.xlsx::Sheet1
- target: Exitus
- positive: 1
- positive_rule: fixed in DATASET_SPECS
- n: 422
- events: 337
- prevalence: 0.7986
- features_used: ['Gender', 'Age_years', 'Endotracheal_intubation', 'Functional_status', 'Asystole', 'Cardiac_arrest_at_home', 'Bystander', 'Time_min', 'Cardiogenic']
- features_dropped: {}
- note: death before hospital arrival (Requena-Morales 2017); prior functional status and arrest-to-EMS time are pre-outcome predictors used in the original article

## depression_heart_failure — included
- table: S2 File.XLSX::hf
- target: Death (1=Yes, 0=No)
- positive: 1
- positive_rule: fixed in DATASET_SPECS
- n: 425
- events: 99
- prevalence: 0.2329
- features_used: ['Age (years)', 'Male (1=Yes, 0=No)', 'PHQ-9', 'Systolic BP (mm Hg)', 'Estimated glomerular filtration rate', 'Ejection fraction (%)', 'Serum sodium (mmol/l)', 'Blood urea nitrogen (mg/dl)', 'Etiology HF(1=Yes, 0=No)', 'Prior diabetes mellitus', 'Elevated level of BNP/NT-BNP (1=Yes, 0=No)']
- features_dropped: {'id': 'identifier', 'Time from HF to Death (days)': 'outcome/time-related name', 'Time from HF to hospitalization (days)': 'outcome/time-related name', 'Hospitalized (1=Yes, 0=No)': 'specified'}
- note: 2-year death (Jani 2016); hospitalization during follow-up is a second outcome: excluded

## lung_cancer_icu — included
- table: pone.0210951.s006.xlsx::Sheet2
- target: Hospital_death
- positive: 1
- positive_rule: fixed in DATASET_SPECS
- n: 264
- events: 160
- prevalence: 0.6061
- features_used: ['sex', 'age', 'Pre_PS', 'PoorPS', 'Histology_coding', 'Stage', 'Distatntmets', 'Surgery', 'WBRT', 'pallRT', 'prev_op', 'prev_rtx', 'prev_gks', 'prev_wbrt', 'prev_ctx', 'prev_ccrt', 'line', 'Refractory', 'Txoff', 'NewlyDx', 'cancer_status', 'Underlying_CV', 'Underlying_stroke', 'underlying_pulmo', 'pulmocomorbid', 'ILD', 'Underlying_DM', 'Underlying_etc', 'other_malignancy', 'smk_hx', 'Pneumonia', 'Sepsis', 'respfailure', 'hemodynamic', 'AKI', 'neurology', 'cardiovasc', 'prevCPR', 'Planned', 'preICUDNR', 'admission_via', 'PrevICUadm', 'Intensivist', 'post2011', 'neutropenia', 'ANC', 'WBC', 'anemia', 'Hb', 'thrombocytopenia', 'PLT', 'hypoalbuminemia', 'Albumin', 'BUN', 'Cr', 'highCr', 'Na', 'K', 'PT_percent', 'aPTT', 'procalcitonin', 'PaO2FiO2', 'SAPS_II', 'APACH_II', 'SOFA_D1']
- features_dropped: {'serial_no': 'not in pre-declared predictor list', 'histology': 'not in pre-declared predictor list', 'Dx': 'not in pre-declared predictor list', 'ICU_cause_intensivist': 'not in pre-declared predictor list', 'Cause_Direct': 'not in pre-declared predictor list', 'Cause_analysis': 'not in pre-declared predictor list', 'MICU_Cause_explain': 'not in pre-declared predictor list', 'Cause_Ca_related': 'not in pre-declared predictor list', 'Cause_Ca_problem': 'not in pre-declared predictor list', 'Cause_Direct_coding_YUJ': 'not in pre-declared predictor list', 'Cause_detail_YUJ': 'not in pre-declared predictor list', 'Ix': 'not in pre-declared predictor list', 'ICUCPR': 'not in pre-declared predictor list', 'Procedure': 'not in pre-declared predictor list', 'procedure_related': 'not in pre-declared predictor list', 'Ninf': 'not in pre-declared predictor list', 'CRP': 'not in pre-declared predictor list', 'bacteremia': 'not in pre-declared predictor list', 'culture_result': 'not in pre-declared predictor list', 'RRT': 'not in pre-declared predictor list', 'RRTetc': 'not in pre-declared predictor list', 'NIV': 'not in pre-declared predictor list', 'HFNC': 'not in pre-declared predictor list', 'MV': 'not in pre-declared predictor list', 'NIVMV': 'not in pre-declared predictor list', 'MV_days': 'not in pre-declared predictor list', 'Vasoactiveagent': 'not in pre-declared predictor list', 'CRRT': 'not in pre-declared predictor list', 'HD': 'not in pre-declared predictor list', 'PCC': 'not in pre-declared predictor list', 'ECMO': 'not in pre-declared predictor list', 'NOVA': 'not in pre-declared predictor list', 'CTxinICU': 'not in pre-declared predictor list', 'Steroid_ICU': 'not in pre-declared predictor list', 'Tracheostomy': 'not in pre-declared predictor list', 'PDT': 'not in pre-declared predictor list', 'BIPAP': 'not in pre-declared predictor list', 'timefromdx': 'not in pre-declared predictor list', 'dx_after_adm': 'not in pre-declared predictor list', 'ICU_LOS': 'not in pre-declared predictor list', 'hospital_LOS': 'not in pre-declared predictor list', 'post_ICU_HD': 'not in pre-declared predictor list', 'post_icu_survival': 'not in pre-declared predictor list', 'post_dc_days': 'not in pre-declared predictor list', 'ICUreadmission': 'not in pre-declared predictor list', 'ICU_Death': 'not in pre-declared predictor list', 'ICU_mortal30': 'not in pre-declared predictor list', 'DischargefromICU': 'not in pre-declared predictor list', 'GW_Death': 'not in pre-declared predictor list', 'Death': 'not in pre-declared predictor list', 'Death_Cause': 'not in pre-declared predictor list', 'Deathcauseexp': 'not in pre-declared predictor list', 'DNR': 'not in pre-declared predictor list', 'postICUCtx': 'not in pre-declared predictor list', 'postICURT': 'not in pre-declared predictor list', 'postICUCPR': 'not in pre-declared predictor list', 'ADM_cause_detailed': 'not in pre-declared predictor list', 'treatment_detailed': 'not in pre-declared predictor list', 'ICU_ADM_Y': 'not in pre-declared predictor list', 'immunotherapy': 'not in pre-declared predictor list', 'performance_ICU_DC': 'not in pre-declared predictor list', 'performance_hospital_DC': 'not in pre-declared predictor list', 'DNR_at_ICU': 'not in pre-declared predictor list'}
- note: hospital death (Song 2019, primary outcome); only variables known at ICU admission (no ICU treatments, no post-ICU or discharge variables)

## clinical_data_14201600 — excluded (n=60, minority=24, features=5)
- table: pone.0248612.s001.xlsx::S1 Table
- target: OS status (1=deceased, 0=living)
- positive: 1
- positive_rule: numeric 0/1 (1 = event unless the name says alive/survival)
- n: 60
- events: 36
- prevalence: 0.6
- features_used: ['sex', 'age [years]', 'progression (1=yes, 0=no)', 'MGMT methylation (1=methylated, 0=unmethylated)', 'cHsp70']
- features_dropped: {'patient ID': 'identifier', 'OS [months]': 'outcome/time-related name', 'PFS [months]': 'outcome/time-related name'}
- note: 

## sepsis_icu_90day — excluded (n=123, minority=44, features=12)
- table: S1_Dataset_Sepsis90DayMortality.xlsx.xlsx::数据3 - 副本
- target: Survival_Status
- positive: 2
- positive_rule: UNCLEAR -> minority class used (CHECK)
- n: 123
- events: 44
- prevalence: 0.3577
- features_used: ['BMI', 'Sex', 'Age', 'SOFA_score', 'APACHEII_Score', 'NUTRIC_Score', 'SMI', 'SMD_HU', 'SMA_cm2', 'SAT_cm2', 'CCI_Score', 'PCT']
- features_dropped: {'Patient_ID': 'identifier', 'Survival_Days': 'outcome/time-related name'}
- note: 

## sepsis_sirs — included
- table: S1_Text.xls::SEPSİS MARKERLARI
- target: Mortality
- positive: 1
- positive_rule: fixed in DATASET_SPECS
- n: 1257
- events: 129
- prevalence: 0.1026
- features_used: ['Age', 'Gender', 'Diagnosis', 'APACHE II', 'SOFA', 'CRP', 'WBCC', 'NeuC', 'LymC', 'EOC', 'NLCR', 'PLTC', 'MPV', 'Group']
- features_dropped: {'LOS-ICU': 'outcome/time-related name'}
- note: ICU mortality (Gucyetmez & Atalan 2016)

## ckd_uae — included
- table: pone.0199920.s002.xlsx::1
- target: EventCKD35
- positive: 1
- positive_rule: fixed in DATASET_SPECS
- n: 491
- events: 56
- prevalence: 0.1141
- features_used: ['Gender', 'AgeBaseline', 'Age.3.categories', 'HistoryDiabetes', 'HistoryCHD', 'HistoryVascular', 'HistorySmoking', 'HistoryHTN', 'HistoryDLD', 'HistoryObesity', 'DLDmeds', 'DMmeds', 'HTNmeds', 'ACEIARB', 'CholesterolBaseline', 'TriglyceridesBaseline', 'HgbA1C', 'CreatnineBaseline', 'eGFRBaseline', 'sBPBaseline', 'dBPBaseline', 'BMIBaseline']
- features_dropped: {'StudyID': 'specified', 'TimeToEventMonths': 'outcome/time-related name'}
- note: CKD stage 3-5 event (Al-Shamsi 2018); StudyID is an identifier

## mesothelioma — no readable table


## heart_failure_chicco — included
- table: uci_519.csv
- target: death_event
- positive: 1
- positive_rule: fixed in DATASET_SPECS
- n: 299
- events: 96
- prevalence: 0.3211
- features_used: ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']
- features_dropped: {'time': 'specified'}
- note: death (Chicco & Jurman 2020); time = follow-up length (outcome-related): excluded

## neuroblastoma_shanghai — no readable table

