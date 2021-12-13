# DNA_Q_assembler
genome assembler using Q learning
1. install https://github.com/kriowloo/gymnome-assembly
2. run "python DNA_Q_assembler.py" to train Q learning agent on the following reads

    reads:  ['CGTTCGGT', 'TTGCGTTC', 'CTTGCGTT', 'ACGCTTGC', 'ATACGCTT', 'AATACGCT', 'AGCAATAC', 'CTAGCAAT', 'ACTAGCAA', 'TACTAGCA']
    
    solution: [9,8,7,6,5,4,3,2,1,0]  ---> TACTAGCAATACGCTTGCGTTCGGT



**current setting (3,000,000 episode , gamma = 0.99 , alpha = 0.8, linear epsilon decay)
