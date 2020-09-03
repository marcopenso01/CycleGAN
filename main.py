
import CycleGAN
from CycleGAN import *

# Create a CycleGAN on GPU 0
myCycleGAN = CycleGAN(0)

input_folder = 'F:\prova\data'
train_A_dir = 'F:\prova\data\trainA'
train_B_dir = 'F:\prova\data\trainB'
output_sample_dir = 'F:\prova\data\output'
model_dir = 'F:\prova\data\models'
batch_size = 10
epochs = 200

read_dicom.load_data(input_folder, force_overwrite=True)

myCycleGAN.train(train_A_dir, train_B_dir, models_dir, batch_size, epochs, output_sample_dir=output_sample_dir, output_sample_channels=1)

for epoch in range(20, 201, 20):
    G_X2Y_dir = '/home/xuagu37/CycleGAN/train_T1_FA/models/G_A2B_weights_epoch_' + str(epoch) + '.hdf5'
    print(G_X2Y_dir)
    test_X_dir = '/home/xuagu37/CycleGAN/data/T1_test.nii.gz'
    synthetic_Y_dir = '/home/xuagu37/CycleGAN/train_T1_FA/synthetic/FA_synthetic_epoch_' + str(epoch) + '.nii.gz'
    normalization_factor_X = 1000
    normalization_factor_Y = 1
    myCycleGAN.synthesize(G_X2Y_dir, test_X_dir, normalization_factor_X, synthetic_Y_dir, normalization_factor_Y)
