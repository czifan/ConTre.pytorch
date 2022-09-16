import os
for name in os.listdir('trainers'):
    if name.startswith('_') or name.startswith('.'): continue
    name = name.replace('.py', '')
    exec(f'from trainers.{name} import *')
