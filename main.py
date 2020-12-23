from CycleGAN import *
import os
import h5py
# Create a CycleGAN on GPU 0
myCycleGAN = CycleGAN('0,1')

input_folder = 'D:\ARTEFATTI\data'
train_A_dir = 'D:\ARTEFATTI\data\TrainA'
train_B_dir = 'D:\ARTEFATTI\data\TrainB'
output_dir = 'D:\ARTEFATTI\data\output\output_sample.png'
model_dir = 'D:\ARTEFATTI\data\models'
batch_size = 6
epochs = 200

def get_mean_value(table1, table2):
    max_values = []
    for i in range(0, len(table1)):
        max_value = table1[i].max()
        max_values.append(max_value)
    for i in range(0, len(table2)):
        max_value = table2[i].max()
        max_values.append(max_value)
    return int(sum(max_values) / len(max_values))


data_file_path = os.path.join(train_A_dir, 'preprocessing', 'train.hdf5')
dt = h5py.File(data_file_path, 'r')
data1 = dt['images_train'][()]
data_file_path = os.path.join(train_B_dir, 'preprocessing', 'train.hdf5')
dt = h5py.File(data_file_path, 'r')
data2 = dt['images_train'][()]
normalization_factor = get_mean_value(data1, data2)
print('normalization value %s' % normalization_factor)

myCycleGAN.train(train_A_dir, train_B_dir, normalization_factor, model_dir, batch_size, epochs,
                 output_sample_dir=output_dir, output_sample_channels=1)


'''
for epoch in range(20, 201, 20):
    G_X2Y_dir = '/home/xuagu37/CycleGAN/train_T1_FA/models/G_A2B_weights_epoch_' + str(epoch) + '.hdf5'
    print(G_X2Y_dir)
    test_X_dir = '/home/xuagu37/CycleGAN/data/T1_test.nii.gz'
    synthetic_Y_dir = '/home/xuagu37/CycleGAN/train_T1_FA/synthetic/FA_synthetic_epoch_' + str(epoch) + '.nii.gz'
    myCycleGAN.synthesize(G_X2Y_dir, test_X_dir, synthetic_Y_dir, normalization_factor)
'''
