import h5py
import json

file_path = 'models/image_caption_model.h5'
with h5py.File(file_path, 'r+') as f:
    if 'model_config' in f.attrs:
        raw_config = f.attrs['model_config']
        if isinstance(raw_config, bytes):
            model_config = json.loads(raw_config.decode('utf-8'))
        else:
            model_config = json.loads(raw_config)
        
        # Walk through the config and remove 'time_major' from LSTM
        fixed = False
        if 'config' in model_config and 'layers' in model_config['config']:
            for layer in model_config['config']['layers']:
                # The functional model's inner layers
                if layer['class_name'] == 'Functional' and 'layers' in layer['config']:
                    for sub_layer in layer['config']['layers']:
                        if sub_layer['class_name'] == 'LSTM':
                            if 'time_major' in sub_layer['config']:
                                del sub_layer['config']['time_major']
                                fixed = True
                # Global layers
                if layer['class_name'] == 'LSTM':
                    if 'time_major' in layer['config']:
                        del layer['config']['time_major']
                        fixed = True
                        
        if fixed:
            f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
            print("Successfully removed time_major from model_config.")
        else:
            print("No time_major found in LSTM layers or already fixed.")
    else:
        print("No model_config found in the HDF5 file.")
