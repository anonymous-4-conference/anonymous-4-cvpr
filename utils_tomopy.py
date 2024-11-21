import h5py
import json

def print_nexus_structure(nexus_file):
    
    with h5py.File(nexus_file, 'r') as f:
        # Function to recursively print the structure of the NeXus file and find the parameter values
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")

        # Print the structure of the file
        f.visititems(print_structure)

        # Check for the plugin parameters in the data fields
        for plugin_id in range(1, 7):  # Assuming there are 6 plugins based on the output
            try:
                data_path = f'/entry/plugin/{plugin_id}/data'
                data = f[data_path][()]
                # Decode bytes to string and parse JSON if needed
                if isinstance(data, bytes):
                    data_str = data.decode('utf-8')
                    try:
                        data_dict = json.loads(data_str)
                        print(f"\nParameters for plugin {plugin_id}:")
                        for key, value in data_dict.items():
                            print(f"{key}: {value}")
                    except json.JSONDecodeError:
                        print(f"Data for plugin {plugin_id} is not in JSON format:\n{data_str}")
                else:
                    print(f"Data for plugin {plugin_id}:\n{data}")
            except KeyError as e:
                print(f"KeyError: {e} - This key does not exist in the file. The structure might be different.")