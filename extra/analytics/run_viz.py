import os

all_files = os.listdir('./extra/analytics')
py_files = [file for file in all_files if (file.endswith('.py') and file != 'run_viz.py')]

for file in py_files:
    os.system('python extra/analytics/' + file)
