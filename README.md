in hyak, open an interactive node (I do 1CPU and 1G) and build the apptainer by running the following commands: 
```
module load apptainer
apptainer build --force /your/hyak/path/putida-bmca.sif docker://janisshin/putidabmca:latest
```
I try to run the container by using `run_singularity.py`
I used scp to do the file transfer

```
scp -r  run_singularity.py user@klone.hyak.uw.edu:/your/hyak/path

```

This involved installing miniconda into hyak and then using Python to run the script.
