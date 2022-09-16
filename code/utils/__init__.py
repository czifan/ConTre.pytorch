import os
for name in os.listdir('utils'):
    if name.startswith('_') or name.startswith('.'): continue
    name = name.replace('.py', '')
    exec(f'from utils.{name} import *')
