def config(shellRcFiles, sourceDir, envName):
    global shellrc
    global srcDir
    sourceShell = ""
    match shellRcFiles:
        case (firstShell, secondShell):
            sourceShell = "source {0:s}\nsource {1:s}\n".format(firstShell, secondShell)
        case str():
            sourceShell = "source {0:s}\n".format(shellrc)
    sourceShell += "conda activate {envName:s}\n".format(envName)
    shellrc = sourceShell

    srcDir = sourceDir


LOCAL_HEADER="""#!/usr/bin/env zsh

{sourcerc:s}

export PATH=$PATH:{bprevealPath:s}/bin
#Get bedtools on the path, this is needed for teak.
export PATH=$PATH:/n/apps/CentOS7/bin

"""

SLURM_HEADER_NOGPU="""#!/usr/bin/env zsh
#SBATCH --job-name {jobName:s}
#SBATCH --ntasks={ntasks:d}
#SBATCH --nodes=1
#SBATCH --mem={mem:d}gb
#SBATCH --time={time:s}
#SBATCH --output={workdir:s}/logs/{jobName:s}_%A_%a.out
#SBATCH --partition=compute
#SBATCH --array=1-{numJobs:d}%10

{sourcerc:s}
export PATH=$PATH:{bprevealPath:s}/bin
#module load bpreveal
module load bedtools
module load meme

"""

def jobsNonGpu(workDir, tasks, jobName, ntasks, mem, time):

    cmd = SLURM_HEADER_NOGPU.format(jobName=jobName, ntasks=ntasks, mem=mem, 
                                    time=time, numJobs=len(tasks), sourcerc=shellrc,
                                    bprevealPath=srcDir, workDir=workDir)
    for i, task in enumerate(tasks):
        cmd += "if [[ ${{SLURM_ARRAY_TASK_ID}} == {0:d} ]] ; then\n".format(i+1)
        cmd += "    {0:s}\n".format(task)
        cmd += "fi\n\n"
    with open(workDir+"/{0:s}.slurm".format(jobName), "w") as fp:
        fp.write(cmd)


def jobsLocal(workDir, tasks, jobName, ntasks, mem, time):
    cmd = LOCAL_HEADER.format(sourcerc=shellrc, bprevealPath=srcDir)
    for task in tasks:
        cmd += "{0:s}\n".format(task)
    with open(workDir+"/{0:s}.zsh".format(jobName), "w") as fp:
        fp.write(cmd)


SLURM_HEADER_NOGPU="""#!/usr/bin/env zsh
#SBATCH --job-name {jobName:s}
#SBATCH --ntasks={ntasks:d}
#SBATCH --nodes=1
#SBATCH --mem={mem:d}gb
#SBATCH --time={time:s}
#SBATCH --output={workdir:s}/logs/{jobName:s}_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres gpu:a100_3g.20gb:1
#SBATCH --array=1-{numJobs:d}%10

{sourcerc:s}
export PATH=$PATH:{bprevealPath:s}/bin
module load bedtools
module load meme

"""



def jobsGpu(workDir, tasks, jobName, ntasks, mem, time):

    cmd = SLURM_HEADER_NOGPU.format(jobName=jobName, ntasks=ntasks, mem=mem, 
                                    time=time, numJobs=len(tasks), sourcerc=shellrc,
                                    bprevealPath=srcDir, workDir=workDir)
    for i, task in enumerate(tasks):
        cmd += "if [[ ${{SLURM_ARRAY_TASK_ID}} == {0:d} ]] ; then\n".format(i+1)
        cmd += "    {0:s}\n".format(task)
        cmd += "fi\n\n"
    with open(workDir+"/{0:s}.slurm".format(jobName), "w") as fp:
        fp.write(cmd)

