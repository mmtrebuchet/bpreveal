def configSlurm(shellRcFiles: list[str] | str, envName: str, workingDirectory: str,
                maxJobs=10) -> dict:
    """shellRcFiles is a list of strings, each one giving the name of
    a .shrc file in your home directory. These will be sourced, in order,
    inside the generated script.
    If shellRcFiles is just a string, it names one and only .shrc file that should be executed.
    envName is a string, and it gives the name for the environment that should be loaded.
    if envName is "ml", then instead of a conda command being used to load bpreveal, the
    module system on cerebro will be used instead.
    If you have not installed bpreveal yourself, you should use "ml" for envName.
    """
    sourceShell = ""
    match shellRcFiles:
        case (firstShell, secondShell):
            sourceShell = "source {0:s}\nsource {1:s}\n".format(firstShell, secondShell)
        case str():
            sourceShell = "source {0:s}\n".format(shellRcFiles)

    if envName == "ml":
        condaString = "module load bpreveal\n"
    else:
        condaString = "conda deactivate\n"
        condaString += "conda activate {envName:s}\n".format(envName=envName)

    return {"workDir": workingDirectory, "sourceShell": sourceShell, "condaString": condaString,
            "gpuType": "a100_3g.20gb", "maxJobs": maxJobs}


LOCAL_HEADER = """#!/usr/bin/env zsh

{sourcerc:s}

#Get bedtools on the path, this is needed for teak.
export PATH=$PATH:/n/apps/CentOS7/bin

{condastring:s}
"""

SLURM_HEADER_NOGPU = """#!/usr/bin/env zsh
#SBATCH --job-name {jobName:s}
#SBATCH --ntasks={ntasks:d}
#SBATCH --nodes=1
#SBATCH --mem={mem:d}gb
#SBATCH --time={time:s}
#SBATCH --output={workDir:s}/logs/{jobName:s}_%A_%a.out
#SBATCH --partition=compute
#SBATCH --array=1-{numJobs:d}%{maxJobs:d}

{sourcerc:s}
#module load bpreveal
module load bedtools
module load meme
module load bedops
module load ucsc
#These are just for non-gpu jobs.
module load bowtie2
module load samtools
module load sratoolkit

{condastring:s}
"""


def jobsNonGpu(config: dict, tasks: list[str], jobName: str,
               ntasks: int, mem: int, time: str, extraHeader=""):
    cmd = SLURM_HEADER_NOGPU.format(jobName=jobName, ntasks=ntasks, mem=mem,
                                    time=time, numJobs=len(tasks), sourcerc=config["sourceShell"],
                                    workDir=config["workDir"], condastring=config["condaString"],
                                    maxJobs=config["maxJobs"])
    cmd += extraHeader + "\n\n"
    for i, task in enumerate(tasks):
        cmd += "if [[ ${{SLURM_ARRAY_TASK_ID}} == {0:d} ]] ; then\n".format(i + 1)
        cmd += "    {0:s}\n".format(task)
        cmd += "fi\n\n"
    outFname = config["workDir"] + "/slurm/{0:s}.slurm".format(jobName)
    with open(outFname, "w") as fp:
        fp.write(cmd)
    return outFname


def jobsLocal(config, tasks, jobName, ntasks, mem, time, extraHeader=""):
    del ntasks
    del mem
    del time
    condaString = config["condaString"]

    if condaString[:6] == "module":
        assert False, "Cannot run local jobs if the configuration specified 'ml'."

    cmd = LOCAL_HEADER.format(sourcerc=config["sourceShell"], condastring=condaString)
    cmd += "\n" + extraHeader + "\n"
    for task in tasks:
        cmd += "{0:s}\n".format(task)
    scriptFname = config["workDir"] + "/slurm/{0:s}.zsh".format(jobName)
    with open(scriptFname, "w") as fp:
        fp.write(cmd)
    import os
    from stat import S_IREAD, S_IWRITE, S_IEXEC, S_IRGRP, S_IROTH
    # Chmod the file to rwxr--r--.
    os.chmod(scriptFname, mode=S_IREAD | S_IWRITE | S_IEXEC | S_IRGRP | S_IROTH)


SLURM_HEADER_GPU = """#!/usr/bin/env zsh
#SBATCH --job-name {jobName:s}
#SBATCH --ntasks={ntasks:d}
#SBATCH --nodes=1
#SBATCH --mem={mem:d}gb
#SBATCH --time={time:s}
#SBATCH --output={workdir:s}/logs/{jobName:s}_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres gpu:{gpuType:s}:1
#SBATCH --array=1-{numJobs:d}%{maxJobs:d}

{sourcerc:s}
module load bedtools
module load meme
module load bedops
module load ucsc
{condastring:s}

"""


def jobsGpu(config, tasks, jobName, ntasks, mem, time, extraHeader=""):

    cmd = SLURM_HEADER_GPU.format(jobName=jobName, ntasks=ntasks, mem=mem,
                                  time=time, numJobs=len(tasks), sourcerc=config["sourceShell"],
                                  workdir=config["workDir"], condastring=config["condaString"],
                                  gpuType=config["gpuType"], maxJobs=config["maxJobs"])
    cmd += extraHeader + "\n\n"

    for i, task in enumerate(tasks):
        cmd += "if [[ ${{SLURM_ARRAY_TASK_ID}} == {0:d} ]] ; then\n".format(i + 1)
        cmd += "    {0:s}\n".format(task)
        cmd += "fi\n\n"
    outFname = config["workDir"] + "/slurm/{0:s}.slurm".format(jobName)
    with open(outFname, "w") as fp:
        fp.write(cmd)
    return outFname


def writeDependencyScript(config, jobspecs, wholeJobName, baseJobId=None):
    """A jobspec is a tuple of (str, [str,str,str,...])
    where the first string is the name of the slurm file to run.
    The strings after it are the names of the slurm files that the
    job needs to finish before it can run (i.e., its dependencies).
    Writes a bash script (NOT A SLURM FILE) that sbatches all of the
    given jobs with dependencies taken into account.
    """
    outFname = config["workDir"] + "/slurm/{0:s}.zsh".format(wholeJobName)
    jobsRemaining = jobspecs[:]
    jobOrder = []
    depsSatisfied = []
    while len(jobsRemaining):
        newRemaining = []
        for js in jobsRemaining:
            job, deps = js
            addJob = True
            for dep in deps:
                if dep not in depsSatisfied:
                    addJob = False
            if addJob:
                depsSatisfied.append(job)
                jobOrder.append(js)
            else:
                newRemaining.append(js)
        assert len(newRemaining) < len(jobsRemaining), "Failed to satisfy jobs: " \
            + str(jobsRemaining) + " with dependencies: " + str(depsSatisfied)
        jobsRemaining = newRemaining

    jobToDepNumber = dict()
    i = 1
    for jobSpec in jobOrder:
        job, deps = jobSpec
        jobToDepNumber[job] = i
        i += 1

    with open(outFname, "w") as fp:
        fp.write("#!/usr/bin/env zsh\n")
        for jobSpec in jobOrder:
            job, deps = jobSpec
            # First, create the dependency string.
            if baseJobId is not None:
                depStr = " --dependency=afterok:{0:d}".format(baseJobId)
            else:
                depStr = ""
            if len(deps):
                depStr = " --dependency=afterok"
                if baseJobId is not None:
                    depStr += ":{0:d}".format(baseJobId)
                for dep in deps:
                    depStr = depStr + ":${{DEP_{0:d}}}".format(jobToDepNumber[dep])
            batchStr = "sbatch" + depStr + " {0:s}".format(job)
            fp.write(batchStr + "\n")
            fp.write(("DEP_{0:d}=$(squeue -u $(whoami) | awk '{{print $1}}' |"
                "sed 's/_.*//'| sort -n | tail -n 1)\n").format(jobToDepNumber[job]))
            fp.write("echo \"job '{0:s}' got dependency ${{DEP_{1:d}}}\"\n".
                     format(job, jobToDepNumber[job]))
