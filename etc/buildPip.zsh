#!/usr/bin/env zsh

BPREVEAL_PATH=..
BPREVEAL_VERSION=5.2.1

runAndCheck() {
    currentCommand=$@
    echo "EXECUTING COMMAND: [[$currentCommand]]"
    eval "$currentCommand"
    errorVal=$?
    if [ $errorVal -eq 0 ]; then
        echo "SUCCESSFULLY EXECUTED: [[$currentCommand]]"
    else
        echo "ERROR DETECTED: Command [[$currentCommand]] on line $BASH_LINENO exited with status $errorVal"
        exit 1
    fi
}

cd $BPREVEAL_PATH

if git status | grep -q "nothing to commit"
then
    echo "Git is clean. Performing uv build."
    runAndCheck uv build

    cd dist/
    runAndCheck auditwheel repair --plat manylinux_2_17_x86_64 bpreveal-${BPREVEAL_VERSION}-cp312-cp312-linux_x86_64.whl

    cp bpreveal-${BPREVEAL_VERSION}.tar.gz wheelhouse

    echo "twine upload --verbose --repository testpypi dist/wheelhouse/*"
else
    echo "Git has unstaged changes."
fi

