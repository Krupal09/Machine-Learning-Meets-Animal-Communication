# Osnabrueck University Grid Computing

## Initial Steps
1. Navigate to the Valid-Until-July-31 folder
2. Run “touch 00-APPLY-FOR-STORAGE/arbitraryfile”
3. A folder with your username will be created in the Valid-Until-July-31 folder. You will also receive an email notification for this. This folder is storage for the project.
4. Navigate to your folder
5. Inside your folder run, “wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh”
6. Run “chmod +x Miniconda3-latest-Linux-x86_64.sh”
7. Run “./Miniconda3-latest-Linux-x86_64.sh”
8. Make sure the path is set to your folder in the SP directory and not outside it
9. Log out, and log in again
10. Run “which conda”. You will be shown a version number. This means you have anaconda. You are mostly set up.

Beyond this point you can install any library using `conda install _arbitrarylibrary_`
Refer to the docs of conda to learn more about its use.

## FAQs

> Q1: What to do in case of running out of space of 5GB in the personal folder? As each home folder has a limit of 5 GB
>
> It is better to not create a conda environment in the home folder but in the scratch drive, which has a limit in TB.

> Q2: How do I submit a command to the queue?
>
> qsub -l mem=4G script.py
> This will start script.py with 4 GB memory.

> Q3: How do I tell if my script which I submitted as a job is done running? What is the current status of my job?
> 
> Type `qstat` in the terminal. If your script is running, it will be in the list shown, otherwise not.

**Q4: Where is the output of my submitted job?**

In your home folder. Type `cd`. You will reach your home folder. Here, you will find 2 files, for the output and the error stream. The name of the file is prefixed with `o-` for output and `e-` for error.

**Q5: The state of my job is Eqw, what should I do?**

Eqw means waiting in the queue in an error state. Type `qstat -j <jobid>` to find out the error. To learn more about error states, go here.

**Q6: What is the job ID of the job I submitted?**

Typing `qstat` in the terminal will show you a table of the jobs you have submitted and their IDs. There is other useful information as well.

Q7: How do I delete a job?
A: `qdel <job_id>`

Q8: The grid tells me I don’t have conda installed but I have installed it. How do I use my conda installation with the grid?
A: qsub -l mem=100M -v PATH trainae.sh
The command above will pass your environment to the grid and your conda install and all the libraries you have will be used.

Q9: I submitted a job and got this error: `/var/lib/gridengine/util/starter.sh: line 41: 29161 Killed           	/usr/bin/cgexec -g freezer,memory,cpuset:${CGPATH} $@`. What is wrong?
A: Your job requires too much memory which cannot be allocated so the grid killed it. Read Question 10.

Q10: My jobs get killed after 90 minutes of run time. Why?
A: There is a “walltime” on the grid. A maximal amount of time is allocated to each type of user of the grid. Most students are in the ge group, where they are allowed (approximately) 25 GB RAM and 90 minutes of running time. If you need more compute power, ask the admins.

Q11: My job has been running for a large amount of time but I have no interim results in the output file. Why?
A: Programs in Linux write to an Input/Output buffer and continue their work, they never make the effort to write to the actual output ‘stream’ (read: file). In order to start seeing interim results, enter `sys.stdio.flush()` after your print statement(s). This will force the program to dump its output to the output file.

Q11: I have received an email on exceeding the quota. What should I do?
A: The quota is exceeded when you have too much content in the home (`~/`) directory. Type `du -sh ~/.[a-zA-Z]*` to view the file size of all the files in the home directory. It is possible you exceeded the quota by either installing conda there, or you installed Python packages and pip has cached the wheel files packages in there.

Q11: How much quota do I have?
A: Run `quota -s` to view your quota.

