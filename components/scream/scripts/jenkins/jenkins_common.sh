#!/bin/bash -xe

# Check if the env var PR_LABELS is defined, and contains something meaningful
IFS=';' read -r -a labels <<< "$PR_LABELS";

# default values
skip_testing=0
test_scripts=0
if [ ${#labels[@]} -gt 0 ]; then
  # We do have some labels. Look for some supported ones.
  for label in "${labels[@]}"
  do
    if [ "$label" == "CI: Integrate Without Testing" ]; then
      skip_testing=1
    elif [ "$label" == "scripts" ]; then
      test_scripts=1
    fi
  done
fi

if [ $skip_testing -eq 0 ]; then
  # User did not request to skip tests. Proceed with testing.
  source ./scream/components/scream/scripts/jenkins/${NODE_NAME}_setup

  git config --global user.email "jenkins@ignore.com"
  git config --global user.name "Jenkins Jenkins"

  SUBMIT="--submit"
  if [ -n "$PULLREQUESTNUM" ]; then
      SUBMIT="" # We don't submit AT runs
  fi

  # The special string "AUTO" makes test-all-scream look for a baseline dir in the machine_specs.py file.
  # IF such dir is not found, then the default (ctest-build/baselines) is used
  BASELINES_DIR=AUTO

  ./scream/components/scream/scripts/gather-all-data "./scripts/test-all-scream --baseline-dir $BASELINES_DIR \$compiler -c EKAT_DISABLE_TPL_WARNINGS=ON -p -i -m \$machine $SUBMIT" -l -m $SCREAM_MACHINE

  # Add a valgrind test for mappy for nightlies
  if [[ -n "$SUBMIT" && "$SCREAM_MACHINE" == "mappy" ]]; then
      ./scream/components/scream/scripts/gather-all-data "./scripts/test-all-scream -t valg --baseline-dir $BASELINES_DIR \$compiler -c EKAT_DISABLE_TPL_WARNINGS=ON -p -i -m \$machine $SUBMIT" -l -m $SCREAM_MACHINE
  fi

  if [ $test_scripts -eq 1 ]; then
      cd scream/components/scream/scripts
      ./scripts-tests -g
      if [ $? -ne 0 ]; then
        exit 1
      fi
      ./scripts-tests -c
      if [ $? -ne 0 ]; then
        exit 1
      fi
      ./scripts-tests -f -m $SCREAM_MACHINE
      if [ $? -ne 0 ]; then
        exit 1
      fi
  fi
else
  echo "Tests were skipped, since the Github label 'CI: Integrate Without Testing' was found.\n"
fi
