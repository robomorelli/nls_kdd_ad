import os
import subprocess
from subprocess import PIPE

try:
    o = subprocess.run(["hostname"],  capture_output=True)
    o = o.stdout
    o = o.strip().decode('utf-8')
except:
    o = subprocess.run(["hostname"],  stdout=PIPE, stderr=PIPE)
    o = o.stdout
    o = o.strip().decode('utf-8')

if 'cnaf' in o:
    try:
        o = subprocess.run(["pwd"],  capture_output=True)
        o = o.stdout
        o = o.strip().decode('utf-8')
        # root = o + '/cell_counting_yellow/'
        root = '/storage/gpfs_maestro/hpc/user/rmorellihpc/nls_kdd_ad/'
    except:
        o = subprocess.run(["pwd"],  stdout=PIPE, stderr=PIPE)
        o = o.stdout
        o = o.strip().decode('utf-8')
        # root = o
        # root = o + '/cell_counting_yellow/'
        root = '/storage/gpfs_maestro/hpc/user/rmorellihpc/nls_kdd_ad/'
else:
    root = ''

cols = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", 'difficulty']

cat_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
cont_cols = [x for x in cols if x not in cat_cols]

model_results_path = root +  'model_results/'
dataset_path = root + 'all_kdd_dataset/'
