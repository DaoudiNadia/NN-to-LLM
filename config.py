"""
Configuration file containing the prompts and some variables for
Nhanes data processing.
"""




PROMPT_TEMPLATE_RAW_NHANES = """
You are given features describing an individual’s dietary supplements and nutrient intake.
Predict the target: 1 if their daily vitamin D intake exceeds 15 µg, 0 if it is 15 µg or less, based on these features.

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""

PROMPT_TEMPLATE_GUIDED_NHANES = """
You are given features describing an individual’s dietary supplements and nutrient intake.
Predict the target: 1 if their daily vitamin D intake exceeds 15 µg, 0 if it is 15 µg or less, based on these features.

Integrated Gradients (IG) indicates each feature’s effect on the prediction: positive increases, negative decreases, larger absolute IG means stronger impact. Numbers in [ ] are the feature’s overall min and max:
Only the listed feature ranges should be used for class reasoning; unlisted values must not be interpreted as evidence for either class:
- reported_serving_size [0.1-9]: 1.3-9 increases predictions (IG=0.0), 0.1-0.5 decreases (IG=-0.7)
- iodine_mcg [4-470]: value of 150 increases predictions (IG=0.1), 4-75 decreases (IG=-0.1)
- zinc_mg [0-70.5]: 16-70.5 increases predictions (IG=0.1), 0-2.8 decreases (IG=-0.1)
- folate_dfe_mcg [26-6673]: 26-553 increases predictions (IG=0.0), 1136-6673 decreases (IG=-0.0)
- folic_acid_mcg [15-3925]: 15-325 increases predictions (IG=0.0), 668-3925 decreases (IG=-0.0)
- total_sugars_g [0.3-16.5]: 2.7-16.5 increases predictions (IG=0.0), 0.3-2.5 decreases (IG=-0.1)
- supplement_days_in_past_month [1-30]: 1-10 decreases (IG=-0.0)
- vitamin_b12_mcg [0.3-1645]: 30.1-1645 increases predictions (IG=0.0), 0.3-3 decreases (IG=-0.1)
- vitamin_c_mg [0-2350]: 100-2350 increases predictions (IG=0.0), 0-30 decreases (IG=-0.1)
- magnesium_mg [0.1-533.3]: 0.1-50 increases predictions (IG=0.1), 105-533.3 decreases (IG=-0.1)
- vitamin_b6_mg [0-150]: 5.1-150 increases predictions (IG=0.1), 0-1.7 decreases (IG=-0.2)
- DSQINIAC [0.7-164.5]: 0.7-14 increases predictions (IG=0.0), 22-164.5 decreases (IG=-0.2)
- dosage_form: ('TABLESPOONS' (IG=-0.6), 'MILLILITERS' (IG=-0.5), 'DROPS' (IG=-0.3), 'Packages' (IG=-0.3), 'TEASPOONS' (IG=-0.3)) decrease
- antacid: ('non_antacid' (IG=0.0)) increase

Top feature pairs showing dependent effects:
- vitamin_b6_mg and zinc_mg (strength: 0.4)
- vitamin_c_mg and zinc_mg (strength: 0.3)
- iodine_mcg and zinc_mg (strength: 0.3)
- reported_serving_size and zinc_mg (strength: 0.2)
- magnesium_mg and vitamin_c_mg (strength: 0.2)

Analysis has shown the following patterns in the dataset:
- 3132 samples with label 1 (80%) have these average or dominant features: (reported_serving_size: 1, iodine_mcg: 118.6, zinc_mg: 8.8, folate_dfe_mcg: 864.6, folic_acid_mcg: 508.6, total_sugars_g: 2.6, supplement_days_in_past_month: 24.6, vitamin_b12_mcg: 24.6, vitamin_c_mg: 84.5, magnesium_mg: 84.4, vitamin_b6_mg: 3.7, DSQINIAC: 16.8); and (dosage_form = TABLETS_CAPSULES, antacid = non_antacid)

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""


PROMPT_TEMPLATE_RAW_CHURN = """
You are given features describing a customer’s banking information.
Predict the target: 1 if the customer churns, 0 if they stay, based on these features.

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""




PROMPT_TEMPLATE_GUIDED_CHURN = """
You are given features describing a customer’s banking information.
Predict the target: 1 if the customer churns, 0 if they stay, based on these features.

Integrated Gradients (IG) indicates each feature’s effect on the prediction: positive increases, negative decreases, larger absolute IG means stronger impact. Numbers in [ ] are the feature’s overall min and max.
Only the listed feature ranges should be used for class reasoning; unlisted values must not be interpreted as evidence for either class:
- CreditScore [350-850]: 668-695 increases predictions (IG=0.0), 774-850 decreases (IG=-0.6)
- Age [18-92]: 46-52 increases predictions (IG=1.3), 18-28 decreases (IG=-1.1)
- Tenure [0-10]: value of 10 decreases (IG=-0.6)
- Balance [0-238387.5]: 83132-97665.6 increases predictions (IG=0.1), value of 0 decreases (IG=-1.6)
- NumOfProducts [1-4]: value of 4 increases predictions (IG=57.7), value of 2 decreases (IG=-0.1)
- EstimatedSalary [11.5-199992.4]: 66799.2-88391.9 increases predictions (IG=0.0), 177235.2-199992.4 decreases (IG=-0.3)
- HasCrCard [0-1]: value of 0 decreases (IG=-0.3)
- IsActiveMember [0-1]: value of 0 increases predictions (IG=0.0), value of 1 decreases (IG=-0.6)
- Geography: ('France' (IG=-0.8), 'Spain' (IG=-0.3), 'Germany' (IG=-0.3)) decrease
- Gender: ('Male' (IG=-0.5), 'Female' (IG=-0.1)) decrease

Top feature pairs showing dependent effects:
- Gender and NumOfProducts (strength: 0.5)
- Gender and Geography (strength: 0.4)
- Geography and NumOfProducts (strength: 0.4)
- Balance and NumOfProducts (strength: 0.3)
- Balance and Geography (strength: 0.2)

Analysis has shown the following patterns in the dataset:
- 1425 samples with label 0 (96%) have these average or dominant features: (CreditScore: 671.4, Age: 38, Tenure: 2, Balance: 14112.5, NumOfProducts: 2, EstimatedSalary: 101485.1, HasCrCard: 1, IsActiveMember: 1); and (Geography = France, Gender = Male)
- 122 samples with label 1 (100%) have these average or dominant features: (CreditScore: 627.3, Age: 47, Tenure: 9, Balance: 105691.2, NumOfProducts: 3, EstimatedSalary: 119765.2, HasCrCard: 1, IsActiveMember: 0); and (Geography = France, Gender = Female)

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""




PROMPT_TEMPLATE_RAW_ADULT = """
You are given features describing an individual’s attributes.
Predict the target: 1 if their income exceeds 50K$ per year, 0 if it is 50K$ or less, based on these features.

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""


PROMPT_TEMPLATE_GUIDED_ADULT = """
You are given features describing an individual’s attributes.
Predict the target: 1 if their income exceeds 50K$ per year, 0 if it is 50K$ or less, based on these features.

Integrated Gradients (IG) indicates each feature’s effect on the prediction: positive increases, negative decreases, larger absolute IG means stronger impact. Numbers in [ ] are the feature’s overall min and max:
Only the listed feature ranges should be used for class reasoning; unlisted values must not be interpreted as evidence for either class:
- age [17-90]: 45-49 increases predictions (IG=0.1), 17-22 decreases (IG=-5.5)
- fnlwgt [12285-1484705]: 12285-72333 decreases (IG=-0.5)
- education-num [1-16]: 14-16 increases predictions (IG=0.6), 1-7 decreases (IG=-1.6)
- capital-gain [0-99999]: 15831-99999 increases predictions (IG=44.2)
- capital-loss [0-4356]: 2267-4356 increases predictions (IG=5.0)
- hours-per-week [1-99]: 46-50 increases predictions (IG=0.5), 1-25 decreases (IG=-3.3)
- workclass: ('Never-worked' (IG=-11.5), 'Without-pay' (IG=-8.5), 'State-gov' (IG=-1.0), 'Self-emp-not-inc' (IG=-0.5), 'Local-gov' (IG=-0.2)) decrease
- education: ('Preschool' (IG=-16.9), '1st-4th' (IG=-5.4), '10th' (IG=-4.1), '5th-6th' (IG=-2.9), '7th-8th' (IG=-2.4)) decrease
- marital-status: ('Married-spouse-absent' (IG=-3.3), 'Separated' (IG=-2.7), 'Never-married' (IG=-2.0), 'Widowed' (IG=-1.7)) decrease; ('Married-AF-spouse' (IG=2.1)) increase
- occupation: ('Priv-house-serv' (IG=-14.2), 'Farming-fishing' (IG=-2.8), 'Handlers-cleaners' (IG=-2.8), 'Machine-op-inspct' (IG=-1.7), 'Other-service' (IG=-1.6)) decrease
- relationship: ('Other-relative' (IG=-3.6), 'Unmarried' (IG=-0.7), 'Own-child' (IG=-0.3)) decrease; ('Wife' (IG=0.9), 'Not-in-family' (IG=0.6)) increase
- race: ('Amer-Indian-Eskimo' (IG=-2.6), 'Other' (IG=-1.7), 'Black' (IG=-1.2)) decrease; ('Asian-Pac-Islander' (IG=0.2), 'White' (IG=0.1)) increase
- sex: ('Female' (IG=-0.8)) decrease; ('Male' (IG=0.3)) increase
- native-country: ('Outlying-US(Guam-USVI-etc)' (IG=-15.6), 'Columbia' (IG=-14.9), 'Holand-Netherlands' (IG=-11.8), 'Dominican-Republic' (IG=-11.2), 'Honduras' (IG=-10.9)) decrease       

Top feature pairs showing dependent effects:
- age and native-country (strength: 0.1)
- age and workclass (strength: 0.1)
- native-country and workclass (strength: 0.1)
- education and native-country (strength: 0.1)
- native-country and occupation (strength: 0.1)

Analysis has shown the following patterns in the dataset:
- 6455 samples with label 0 (94%) have these average or dominant features: (age: 36.8, fnlwgt: 202390.3, education-num: 9.6, capital-gain: 167.9, capital-loss: 80, hours-per-week: 39); and (workclass = Private, education = HS-grad, marital-status = Never-married, occupation = Adm-clerical, relationship = Not-in-family, race = White, sex = Male, native-country = United-States)
- 862 samples with label 1 (97%) have these average or dominant features: (age: 47.2, fnlwgt: 202402.7, education-num: 12.4, capital-gain: 15029.4, capital-loss: 0, hours-per-week: 48.1); and (workclass = Private, education = Bachelors, marital-status = Married-civ-spouse, occupation = Exec-managerial, relationship = Husband, race = White, sex = Male, native-country = United-States)

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""




PROMPT_TEMPLATE_RAW_CREDITG = """
You are given features describing an individual’s attributes.
Predict the target: 1 if their class is "good" (creditworthy individuals), 0 if the class is "bad" (not creditworthy), based on these features.

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""




PROMPT_TEMPLATE_GUIDED_CREDITG = """
You are given features describing an individual’s attributes.
Predict the target: 1 if their class is "good" (creditworthy individuals), 0 if the class is "bad" (not creditworthy), based on these features.

Integrated Gradients (IG) indicates each feature’s effect on the prediction: positive increases, negative decreases, larger absolute IG means stronger impact. Numbers in [ ] are the feature’s overall min and max:
Only the listed feature ranges should be used for class reasoning; unlisted values must not be interpreted as evidence for either class:
- duration [4-72]: 4-9 increases predictions (IG=4.9), 39-72 decreases (IG=-5.9)
- credit_amount [250-18424]: 1288-1552 increases predictions (IG=0.4), 6999-18424 decreases (IG=-9.0)
- installment_commitment [1-4]: value of 1 increases predictions (IG=6.4), value of 4 decreases (IG=-1.2)
- residence_since [1-4]: value of 1 increases predictions (IG=6.9)
- age [19-75]: 51-75 increases predictions (IG=4.3), 19-24 decreases (IG=-3.7)
- existing_credits [1-4]: value of 3 increases predictions (IG=1.9), value of 4 decreases (IG=-0.2)
- num_dependents [1-2]: value of 2 increases predictions (IG=2.8), value of 1 decreases (IG=-0.1)
- checking_status: ('<0' (IG=-4.1), '0<=X<200' (IG=-2.3)) decrease; ('no checking' (IG=6.9), '>=200' (IG=1.2)) increase
- credit_history: ('no credits/all paid' (IG=-5.3), 'all paid' (IG=-5.1), 'existing paid' (IG=-0.8)) decrease; ('critical/other existing credit' (IG=5.2), 'delayed previously' (IG=2.6)) increase
- purpose: ('new car' (IG=-5.1), 'other' (IG=-4.1)) decrease; ('used car' (IG=15.5), 'retraining' (IG=12.5), 'domestic appliance' (IG=3.7)) increase
- savings_status: ('500<=X<1000' (IG=-2.7), '<100' (IG=-0.7)) decrease; ('>=1000' (IG=10.1), 'no known savings' (IG=2.6), '100<=X<500' (IG=2.5)) increase
- employment: ('unemployed' (IG=-4.1), '<1' (IG=-1.7)) decrease; ('4<=X<7' (IG=5.1), '>=7' (IG=1.4), '1<=X<4' (IG=1.1)) increase
- personal_status: ('male div/sep' (IG=-3.3)) decrease; ('male mar/wid' (IG=4.2), 'male single' (IG=0.6), 'female div/dep/mar' (IG=0.4)) increase
- other_parties: ('co applicant' (IG=-2.6), 'none' (IG=-0.6)) decrease; ('guarantor' (IG=10.1)) increase
- property_magnitude: ('no known property' (IG=-1.8)) decrease; ('real estate' (IG=1.1), 'life insurance' (IG=0.6), 'car' (IG=0.6)) increase
- other_payment_plans: ('bank' (IG=-1.5), 'stores' (IG=-1.1)) decrease; ('none' (IG=1.3)) increase
- housing: ('for free' (IG=-1.6), 'rent' (IG=-1.3)) decrease; ('own' (IG=1.8)) increase
- job: ('unskilled resident' (IG=-2.6), 'skilled' (IG=-0.5)) decrease; ('unemp/unskilled non res' (IG=8.9), 'high qualif/self emp/mgmt' (IG=3.6)) increase
- own_telephone: ('none' (IG=-1.1)) decrease; ('yes' (IG=2.0)) increase
- foreign_worker: ('yes' (IG=-1.2)) decrease; ('no' (IG=14.3)) increase

Top feature pairs showing dependent effects:
- other_parties and residence_since (strength: 0.2)
- employment and residence_since (strength: 0.2)
- job and residence_since (strength: 0.1)
- checking_status and other_payment_plans (strength: 0.1)
- credit_amount and duration (strength: 0.1)

Analysis has shown the following patterns in the dataset:
- 108 samples with label 0 (81%) have these average or dominant features: (duration: 21, credit_amount: 2663.1, installment_commitment: 4, residence_since: 2, age: 28.9, existing_credits: 1, num_dependents: 1); and (checking_status = <0, credit_history = existing paid, purpose = furniture/equipment, savings_status = <100, employment = 1<=X<4, personal_status = male single, other_parties = none, property_magnitude = life insurance, other_payment_plans = none, housing = own, job = skilled, own_telephone = none, foreign_worker = yes)
- 130 samples with label 1 (96%) have these average or dominant features: (duration: 21, credit_amount: 3420.9, installment_commitment: 4, residence_since: 4, age: 39.1, existing_credits: 1, num_dependents: 1); and (checking_status = no checking, credit_history = existing paid, purpose = radio/tv, savings_status = <100, employment = >=7, personal_status = male single, other_parties = none, property_magnitude = car, other_payment_plans = none, housing = own, job = skilled, own_telephone = yes, foreign_worker = yes)

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""




PROMPT_TEMPLATE_RAW_HIGGSMALL = """
You are given features describing simulated high-energy physics collision events.
Predict the target: 1 if the event corresponds to a Higgs boson signal, 0 if it is background noise, based on these features.

Here are the features:

{sample}

Return only the prediction (1 or 0). No comments or extra text.
"""

PROMPT_TEMPLATE_GUIDED_HIGGSMALL = """
You are given features describing simulated high-energy physics collision events.
Predict the target: 1 if the event corresponds to a Higgs boson signal, 0 if it is background noise, based on these features.

Integrated Gradients (IG) indicates each feature’s effect on the prediction: positive increases, negative decreases, larger absolute IG means stronger impact. Numbers in [ ] are the feature’s overall min and max. 
Only the listed feature ranges should be used for class reasoning; unlisted values must not be interpreted as evidence for either class:
- m_bb [0.04-11.99]: 0.75-0.83 increases predictions (IG=0.3), 1.54-11.99 decreases (IG=-2.1)
- m_wwbb [0.35-6.01]: 0.35-0.69 increases predictions (IG=2.4), 1.29-6.01 decreases (IG=-1.2)
- missing_energy_magnitude [0.0-7.74]: 0.37-0.53 increases predictions (IG=0.0), 1.68-7.74 decreases (IG=-0.5)
- m_wbb [0.3-7.31]: 1.4-7.31 increases predictions (IG=0.8), 0.3-0.71 decreases (IG=-2.7)
- jet1pt [0.13-7.06]: 1.49-7.06 increases predictions (IG=0.8), 0.13-0.53 decreases (IG=-0.4)
- lepton_pT [0.27-7.8]: 0.67-0.79 increases predictions (IG=0.0), 1.67-7.8 decreases (IG=-0.4)
- jet2b-tag [0.0-2.21]: value of 0.0 increases predictions (IG=0.0), value of 2.21 decreases (IG=-0.0)
- jet4pt [0.36-7.51]: 1.59-7.51 increases predictions (IG=0.1), 0.36-0.48 decreases (IG=-0.1)
- m_jjj [0.3-10.03]: 1.27-10.03 increases predictions (IG=0.6), 0.88-0.92 decreases (IG=-0.3)
- jet3b-tag [0.0-2.54]: value of 0.0 increases predictions (IG=0.1), value of 2.54 decreases (IG=-0.1)
- m_jlv [0.31-7.44]: 0.31-0.65 increases predictions (IG=0.5), 0.81-0.88 decreases (IG=-0.2)
- jet2pt [0.18-8.28]: 1.53-8.28 increases predictions (IG=0.1), 0.63-0.73 decreases (IG=-0.0)
- jet4b-tag [0.0-3.1]: value of 0.0 increases predictions (IG=0.1), value of 3.1 decreases (IG=-0.1)
- jet3pt [0.26-8.5]: 1.55-8.5 increases predictions (IG=0.2), 0.26-0.5 decreases (IG=-0.0)
- m_jj [0.11-18.42]: 1.06-1.39 increases predictions (IG=0.0), 1.39-18.42 decreases (IG=-0.2)
- m_lv [0.13-4.56]: 1.18-4.56 increases predictions (IG=0.2), 0.13-0.98 decreases (IG=-0.0)


Top feature pairs showing dependent effects:
- m_bb and m_wbb (strength: 0.4)
- m_wbb and m_wwbb (strength: 0.4)
- m_jj and m_jjj (strength: 0.2)
- m_jlv and m_wbb (strength: 0.1)
- m_jjj and m_jlv (strength: 0.1)

Analysis has shown the following patterns in the dataset:
- 995 samples with label 0 (93%) have these average or dominant features: (m_bb: 2.69, m_wwbb: 1.7, missing_energy_magnitude: 1.44, m_wbb: 1.89, jet1pt: 1.72, lepton_pT: 1.6, jet2b-tag: 2.21, jet4pt: 1.21, m_jjj: 0.99, jet3b-tag: 0.0, m_jlv: 1.02, jet2pt: 1.77, jet4b-tag: 0.0, jet3pt: 1.39, m_jj: 0.88, m_lv: 1.05)

Here are the features:

{sample}

Return the prediction (1 or 0).  No comments or extra text.
"""




mapping_DSD122U = {
    1: "TABLETS_CAPSULES",
    2: "DROPPER",
    3: "DROPS",
    6: "LOZENGES",
    7: "MILLILITERS",
    11: "TABLESPOONS",
    12: "TEASPOONS",
    13: "WAFERS",
    16: "GRAMS",
    19: "SPRAYS",
    20: "CHEWS",
    21: "SCOOP",
    23: "CAPFUL",
    27: "OUNCES",
    28: "Packages",
    32: "BOTTLE",
    41: "PIECES",
}

mapping_DSDANTA = {
    0: "non_antacid",
    1: "antacid_with_supplement",
    2: "antacid_with_medication"
}


feat_mapping = {
    'DSDACTSS': 'reported_serving_size',
    'DSQISUGR': 'total_sugars_g',
    'DSQIVB6': 'vitamin_b6_mg',
    'DSQIFA': 'folic_acid_mcg',
    'DSQIFDFE': 'folate_dfe_mcg',
    'DSQIVB12': 'vitamin_b12_mcg',
    'DSQIVC': 'vitamin_c_mg',
    'DSQIMAGN': 'magnesium_mg',
    'DSQIZINC': 'zinc_mg',
    'DSQIIODI': 'iodine_mcg',
    'DSD103': 'supplement_days_in_past_month',
    'DSD122U': 'dosage_form',
    'DSDANTA': 'antacid'
}


NUM_FEATURES = ['DSDACTSS', 'DSQIKCAL', 'DSQIPROT', 'DSQICARB', 'DSQISUGR',
                'DSQIFIBE', 'DSQITFAT', 'DSQISFAT', 'DSQIMFAT', 'DSQIPFAT',
                'DSQICHOL', 'DSQILYCO', 'DSQILZ', 'DSQIVB1', 'DSQIVB2',
                'DSQINIAC', 'DSQIVB6', 'DSQIFA', 'DSQIFDFE', 'DSQICHL',
                'DSQIVB12', 'DSQIVC', 'DSQIVK', 'DSQICALC',
                'DSQIPHOS', 'DSQIMAGN', 'DSQIIRON', 'DSQIZINC', 'DSQICOPP',
                'DSQISODI', 'DSQIPOTA', 'DSQISELE', 'DSQICAFF', 'DSQIIODI']

CATEG_FEATURES = ['DSD122U', 'DSDANTA']
NUM_TO_FILTER = ['DSD103', 'DSD122Q']

special_missing = {
    'DSD103': [7777, 9999],
    'DSD122Q': [777777, 999999],
    'DSD122U': [77, 99],
}
