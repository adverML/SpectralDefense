import os
import numpy as np
import csv
import pandas as pd

pm = r"$\pm$"
print(pm)

np.set_printoptions(formatter = {
    'float': lambda x: "{0:0.1f}".format(x)
 })


output_folder = "var_gauss"

FILE_ENDING = ".csv"
pm = r"$\pm$"
main_path = "."
# datasets=['cif10vgg', 'cif100vgg']
# datasets=['cif10', 'cif100']
# datasets=['imagenet32', 'imagenet64','imagenet128']
# datasets=['imagenet128']

# datasets=['celebaHQ32', 'celebaHQ64','celebaHQ128']
# datasets=['celebaHQ32']
# datasets=['cif100rn34']
# datasets=['cif10rn34', 'cif100rn34']

# datasets=['imagenet32', 'imagenet64', 'cif10_rb', 'imagenet']
# datasets=['cif10', 'cif100', 'cif10vgg', 'cif100vgg', 'imagenet32', 'imagenet64', 'imagenet128', 'celebaHQ32', 'celebaHQ64', 'cif10_rb', 'imagenet']
datasets=['celebaHQ128']

columns = ['asr', 'auc', 'fnr', 'asrd']
new_cols = ['Unnamed: 0', 'asr', 'auc', 'fnr', 'asrd']

bases = []
indexes = []


runs = 3



for dataset in datasets:
    print(dataset)
    d_frame = []
    np_arr = []

    for idx_run in range(1, runs+1):
        table = pd.read_csv(os.path.join(main_path, "run_gauss_" + str(idx_run), dataset+ FILE_ENDING))
        # import pdb; pdb.set_trace()
        table.pop('acc'); table.pop('pre'); table.pop('tpr'); table.pop('f1')
        cols = table.columns.values
        
        table = table.reindex(columns=new_cols)
        d_frame.append( table )
        arr1 = d_frame[idx_run - 1].to_numpy()
        np_arr.append(np.delete(arr1, [0], 1).astype('float32'))

    print( "len(d_frame)", len(d_frame) )
    print( "same shape  ", np_arr[0].shape == np_arr[1].shape == np_arr[2].shape )
    # import pdb; pdb.set_trace()
    
    i, j = np_arr[0].shape
    result = np.zeros_like(np_arr[0])

    for row in range(i):
        for col in range(j):
            # print( "row,col", row, col)
            variance = np.var( np.array([np_arr[0][row, col], np_arr[1][row, col], np_arr[2][row, col] ]))
                                
            if variance > 500:
                print(np_arr[0][row, col], np_arr[1][row, col], np_arr[2][row, col])
                print("varrrrrrrrrrrrrr", variance)
            result[row, col] = variance
    print(np.around(result,1))
    
    str_var = np.around(result,1).astype(str)
    base = np_arr[0].astype(str)
    
    for row in range(i):
        for col in range(j):
            base[row,col] = base[row,col] + pm + str_var[row,col]
    
    bases.append(base)
    indexes.append(d_frame[0].iloc[:,0].to_numpy())
    
    df = pd.DataFrame(base, columns=columns)
    df.insert(0, "0", d_frame[0].iloc[:,0], True)

    output_path = os.path.join( output_folder, dataset + "_var.csv")
    print ( "save_to: ", output_path)
    df.to_csv(output_path , index=True)


# import pdb; pdb.set_trace()
bases = np.vstack(bases)
indexes = np.hstack(indexes)
# import pdb; pdb.set_trace()
df_all = pd.DataFrame(np.insert(bases, 0, indexes, axis=1), columns=new_cols)
df_all['Unnamed: 0'] = indexes.astype(str)
output_path = os.path.join( output_folder,  "all_var.csv")
print ( "save_to: ", output_path)
df_all.to_csv(output_path , index=True)
