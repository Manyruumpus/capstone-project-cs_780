command to prepare the submission and then submit the submission.zip file on the codebench 

# In CS780-OBELIX directory
cp weights_lvl2_eXXXX.pth weights.pth  # or rename appropriately

python -m py_compile ./agent.py
python -c "import numpy as np, agent; print(agent.policy(np.zeros(18,dtype=np.float32)))"

Compress-Archive -Path .\agent.py, .\weights.pth -DestinationPath .\submission.zip -Force