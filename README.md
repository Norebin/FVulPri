** How to use
1. Data: Refer to the contents in the dataset folder, download the data and perform preliminary processing. The related functions after processing are located in funcs.
2. ACVSS: Refer to ser.yml to configure the required operating environment and run BERT to learn the extracted commit message and vulnerable description.
3. Model: Refer to py36.yml to configure the required operating environment and perform graph representation learning on the generated PDG (some related data are stored in the LR directory).
4. FVulPri: Refer to the content in SCORE to score the vulnerabilities in the dataset, where the EXScore score related content is under exploit-rate.
