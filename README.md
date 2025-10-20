# README

This is a readme document for the paper "Machine Judges Reduce Sentencing Bias". 

## Guidance For Data

We illustrate the English translation for our data's factors. 

| 预备        | 未遂    | 中止  | 较大             | 巨大 | 特别巨大        | 凶器          | 多次         | 流窜    | 扒窃      | 入户     | 系自首         | 立功                | 系坦白     | 如实供述                   | 自愿认罪           | 认罪认罚      | 累犯       | 前科            | 未成年   | 老年人   | 残疾     | 精神病         | 谅解          | 和解           | 赔偿        | 黑恶势力          | 法官原始刑期     | 预测刑期              |
| ----------- | ------- | ----- | ---------------- | ---- | --------------- | ------------- | ------------ | ------- | --------- | -------- | -------------- | ------------------- | ---------- | -------------------------- | ------------------ | ------------- | ---------- | --------------- | -------- | -------- | -------- | -------------- | ------------- | -------------- | ----------- | ----------------- | ---------------- | --------------------- |
| Preparation | Attempt | Abort | Relatively large | Huge | Especially huge | Take a weapon | Repeat crime | Roaming | Snatching | Burglary | Self-surrender | Meritorious service | Confession | Make a truthful confession | Admission of guilt | Plea leniency | Recidivism | Criminal record | Juvenile | Old-aged | Disabled | Mental illness | Understanding | Reconciliation | Restitution | Underworld forces | Judge sentencing | Prediction Sentencing |

## Code Files

### sentencing_model.py

Machine learning model for theft crime sentencing prediction using legal features.

**Features**: Extracts 28 binary legal factors from court documents and theft amount for regression modeling.

**Model**: XGBoost regressor optimized for CAIL2018 evaluation metric.

**Usage**:
```bash
python sentencing_model.py --input_file data.csv --output_dir output/
```

Required columns: `全文` (full text), `刑期` (sentence), `盗窃金额` (theft amount), `序号` (case ID).

### similarity.Rmd & Empirical Elec.Rmd

Statistical analysis code for similarity measurement and empirical evaluation.

## Authorship Declaration

The similarity and Empirical Elec code is written by Mingyang Chen; The rest of code is written by Zhipeng Wu. 

If you want to use our data to conduct research, please contact Mingyang Chen at mc55649@um.edu.mo
