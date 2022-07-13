"""
This file shows an example of how to train a network 100 epochs with:
- classifier head with 3 fully connected layers with 1080, 540 and 1 neurons each respectively
- validation split of 20%
- Data augmentation with Rotation (20 degrees) and Flips (Horizontal and Vertical)

After the training is complete the network will then be used to classify the
test dataset in order to test its performance.
"""

import subprocess

base_network = 'mobilenetv2'
val_split = 0.2
base_test_idx = 1000

# Allows a previously trained weights to be re-loaded
checkpoint_filepath = f'{base_network}_test98'

idx = base_test_idx + 0
save_checkpoint_filepath = f'{base_network}_test{idx}'
with open(f'test{idx}.log', 'w') as custom_nohup:
    if os.name == "nt":
        args = ['python', 'classifier.py',
            't',
            f'{checkpoint_filepath}',
            f'--savecheckpoint', f'{save_checkpoint_filepath}',
            f'--dataaug',
            '--val_split', f'{val_split}',
            '--customclassifier', '1080,540',
            '--numepochs', '100',
            ]
    else:
        args = ['nohup', 'python', 'classifier.py',
            't',
            f'{checkpoint_filepath}',
            f'--savecheckpoint', f'{save_checkpoint_filepath}',
            f'--dataaug',
            '--val_split', f'{val_split}',
            '--customclassifier', '1080,540',
            '--numepochs', '100',
            ]

    
    p = subprocess.Popen(args, stdout=custom_nohup, stderr=custom_nohup)
    p.wait()

    args = f'python classifier.py p {save_checkpoint_filepath} --test --customclassifier 1080,540'.split(' ')
    p = subprocess.Popen(args)
    p.wait()